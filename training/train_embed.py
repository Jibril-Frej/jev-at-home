"""Post-train an embedding model into a bi-encoder decision model (E; jevtrain/embed_model.py).

Same data, losses (ce on gold + KL to the stored teacher distribution), distractor augmentation, dev metrics and
temperature fitting as train_decision.py; no FLAN replay (the embedding encoder has no MLM head).

usage: train_embed.py --train data/jevhome/train.jsonl --dev data/jevhome/dev.jsonl
       --model ibm-granite/granite-embedding-small-english-r2 --out checkpoints/E-s0 [...]
"""
from __future__ import annotations

import argparse
import json
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from jevtrain.decision import Item, batches_by_budget, decision_loss, distract, distractor_pool, evaluate_embed, fit_temperatures, summarize
from jevtrain.embed_model import EmbedDecisionModel, build_batch, model_pooling, texts_for
from jevtrain.flan import lr_at
from jevtrain.heads import QTYPES
from jevtrain.records import read_jsonl


def row_texts(it, prng=None, pool=None, distract_frac=0.0):
    req = it.req
    if pool and prng is not None and prng.random() < distract_frac:
        req = distract(req, it, pool, prng)
    return texts_for(req, it.qtype)


def tensors_for(items, idx):
    kmax = max(items[i].k for i in idx)
    gold = torch.full((len(idx),), -1, dtype=torch.long)
    target = torch.zeros((len(idx), kmax))
    qtype = torch.zeros((len(idx),), dtype=torch.long)
    for r, i in enumerate(idx):
        it = items[i]
        if it.gold is not None:
            gold[r] = it.gold
        target[r, : it.k] = torch.tensor(it.target, dtype=torch.float32)
        qtype[r] = QTYPES[it.qtype]
    return {"gold": gold, "target": target, "qtype": qtype}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--dev", nargs="+", required=True)
    ap.add_argument("--calib-dev", nargs="*", default=None)
    ap.add_argument("--model", default="ibm-granite/granite-embedding-small-english-r2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--loss", default="ce,kl")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--lr-head", type=float, default=5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--distract-frac", type=float, default=0.3)
    ap.add_argument("--budget", type=int, default=20000, help="approximate tokens per batch (states + options)")
    ap.add_argument("--pooling", default="auto", choices=["auto", "cls", "mean"], help="auto = the model's sentence-transformers pooling")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    prng = random.Random(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = device.type == "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    losses = args.loss.split(",")

    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    enc = AutoModel.from_pretrained(args.model, dtype=torch.float32, attn_implementation="sdpa")
    pooling = model_pooling(args.model) if args.pooling == "auto" else args.pooling
    print(f"[embed] pooling={pooling}", flush=True)
    model = EmbedDecisionModel(enc, pooling=pooling).to(device).train()

    rows = [r for p in args.train for r in read_jsonl(p)]
    dev_rows = [r for p in args.dev for r in read_jsonl(p)]
    if args.limit:
        rows, dev_rows = rows[: args.limit], dev_rows[: max(8, args.limit // 5)]
    items = [it for it in (Item(r) for r in rows) if it.gold is not None or it.row.get("teacher")]
    dev_items = [Item(r) for r in dev_rows]
    pool = distractor_pool(items, prng) if args.distract_frac > 0 else []
    print(f"[embed] {len(items)} train / {len(dev_items)} dev, model={args.model} "
          f"params={sum(p.numel() for p in model.parameters()) / 1e6:.0f}M device={device}", flush=True)

    new = set(id(p) for p in model.new_parameters())
    groups = []
    for lr, sel in ((args.lr, lambda p: id(p) not in new), (args.lr_head, lambda p: id(p) in new)):
        dec = [p for p in model.parameters() if sel(p) and p.ndim >= 2]
        nod = [p for p in model.parameters() if sel(p) and p.ndim < 2]
        groups += [{"params": dec, "weight_decay": args.weight_decay, "peak": lr}, {"params": nod, "weight_decay": 0.0, "peak": lr}]
    opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.98), eps=1e-6, fused=amp)

    # per-epoch texts (distractors drawn once per epoch)
    epochs = []
    for ep in range(args.epochs):
        texts = [row_texts(it, prng, pool, args.distract_frac) for it in items]
        epochs.append((texts, batches_by_budget(texts, args.budget, rng)))
    total = sum(len(b) for _, b in epochs)
    if args.max_steps:
        total = min(total, args.max_steps)
    warmup = int(args.warmup_frac * total)
    print(f"[embed] {total} steps, warmup {warmup}", flush=True)
    log = open(out / "train_log.jsonl", "a")
    (out / "train_args.json").write_text(json.dumps(vars(args), indent=1))
    step, t0, run, run_n = 0, time.time(), defaultdict(float), 0
    for ep, (texts, batches) in enumerate(epochs):
        for idx in batches:
            if step >= total:
                break
            b = build_batch(tok, [texts[i] for i in idx], device)
            meta = {k: v.to(device) for k, v in tensors_for(items, idx).items()}
            lr = lr_at(step, total, warmup, 1.0)
            for g in opt.param_groups:
                g["lr"] = lr * g["peak"]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits, valid = model(b)
            tgt = torch.zeros_like(logits)
            tgt[:, : meta["target"].shape[1]] = meta["target"]
            loss, parts = decision_loss(logits.float(), valid, {"gold": meta["gold"], "target": tgt}, losses)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            with torch.no_grad():
                has = meta["gold"] >= 0
                acc = float((logits.argmax(-1)[has] == meta["gold"][has]).float().mean()) if bool(has.any()) else 0.0
            run["loss"] += float(loss); run["acc"] += acc; run_n += 1
            for k, v in parts.items():
                run[k] += v
            if step % args.log_every == 0 or step == total:
                el = time.time() - t0
                rec = {"step": step, "epoch": ep, **{k: round(v / run_n, 4) for k, v in run.items()}, "lr": lr * args.lr,
                       "grad_norm": round(float(gn), 3), "rows": len(idx), "elapsed_s": round(el, 1), "eta_min": round(el / step * (total - step) / 60, 1)}
                print("[embed] " + json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n"); log.flush()
                run, run_n = defaultdict(float), 0

    per_item = evaluate_embed(model, tok, dev_items, device, amp)
    s = summarize(per_item)
    if args.calib_dev:
        calib = [it for it in (Item(r) for p in args.calib_dev for r in read_jsonl(p)) if it.gold is not None]
        temps = fit_temperatures(evaluate_embed(model, tok, calib, device, amp), t_min=0.05)
        temps["_fitted_on"] = args.calib_dev
    else:
        temps = fit_temperatures(per_item, t_min=0.05)
    model.save(out)
    tok.save_pretrained(out)
    (out / "calibration.json").write_text(json.dumps(temps, indent=1))
    final = {"steps": step, "elapsed_s": time.time() - t0, "n_train": len(items), "dev": s, "calibration": temps,
             "head": "emb", "variant": "bi", "loss": losses, "model": args.model}
    (out / "train_summary.json").write_text(json.dumps(final, indent=1))
    print("[embed] done dev=" + json.dumps({k: round(s[k], 4) for k in ("accuracy", "nll", "brier", "ece")}), flush=True)


if __name__ == "__main__":
    main()
