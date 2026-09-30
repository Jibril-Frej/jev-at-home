"""Post-train a ModernBERT-style masked LM into a typed-decision cross-encoder (Ettin-1B, L, B).

Head: jevtrain/heads.py (one logit per option, read at the option's [MASK] marker).
Losses: ce (gold option) + kl (teacher probabilities). Augmentation: choice options shuffled from epoch 1 on
(--perm-aug), an unrelated sentence added to a fraction of the states (--distract-frac). Optional FLAN replay
(--replay, --replay-frac of the steps are ATP batches through the MLM head, a regulariser).
Ends with dev metrics and a per-(type, K-bucket) temperature (calibration.json, fitted on --calib-dev).

usage: train_decision.py --train data/jevhome/train.jsonl --dev data/jevhome/dev.jsonl
       --model answerdotai/ModernBERT-Large-Instruct --out checkpoints/L-s0 [...]
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

from jevtrain.decision import Item, collate, decision_loss, distractor_pool, evaluate_cross, fit_temperatures, forward_options, render, summarize
from jevtrain.flan import Pool, collate as flan_collate, lr_at, make_batches
from jevtrain.heads import MarkerHead
from jevtrain.records import read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", nargs="+", required=True)
    ap.add_argument("--dev", nargs="+", required=True)
    ap.add_argument("--calib-dev", nargs="*", default=None, help="rows to fit the temperature on (default: --dev)")
    ap.add_argument("--model", default="answerdotai/ModernBERT-Large-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--loss", default="ce,kl", help="comma list of ce,kl")
    ap.add_argument("--perm-aug", choices=["none", "choice", "all"], default="choice",
                    help="random option order per epoch (epoch 0 keeps canonical order)")
    ap.add_argument("--distract-frac", type=float, default=0.0,
                    help="fraction of rows per epoch that get an unrelated sentence appended/prepended to the state (label unchanged)")
    ap.add_argument("--replay", default=None, help="FLAN pool dir (prepare_flan.py output); skipped if missing")
    ap.add_argument("--replay-frac", type=float, default=0.2)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--tokens-per-batch", type=int, default=16384)
    ap.add_argument("--grad-ckpt", action="store_true", help="gradient checkpointing in the encoder (memory)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=3e-5, help="encoder + MLM head")
    ap.add_argument("--lr-head", type=float, default=2e-4, help="new head parameters")
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="debug: first N train rows")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    prng = random.Random(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = device.type == "cuda"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    losses = [x for x in args.loss.split(",") if x]

    from transformers import AutoModelForMaskedLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    mlm = AutoModelForMaskedLM.from_pretrained(args.model, dtype=torch.float32, attn_implementation="sdpa")
    mlm.config.sparse_prediction = False
    mlm.sparse_prediction = False
    if args.grad_ckpt:
        mlm.model.gradient_checkpointing_enable()
    head_model = MarkerHead(mlm).to(device).train()

    train_rows = [r for p in args.train for r in read_jsonl(p)]
    dev_rows = [r for p in args.dev for r in read_jsonl(p)]
    if args.limit:
        train_rows, dev_rows = train_rows[: args.limit], dev_rows[: max(8, args.limit // 5)]
    items = [Item(r) for r in train_rows]
    items = [it for it in items if it.gold is not None or it.row.get("teacher")]
    dev_items = [Item(r) for r in dev_rows]
    print(f"[train] {len(items)} train rows, {len(dev_items)} dev rows, loss={losses} model={args.model} "
          f"params={sum(p.numel() for p in head_model.parameters()) / 1e6:.0f}M device={device}", flush=True)

    replay = None
    if args.replay and (Path(args.replay) / "train_pool.tokens.npy").exists():
        replay = Pool(str(Path(args.replay) / "train_pool"))
        print(f"[train] replay pool {args.replay}: {len(replay):,} examples, frac {args.replay_frac}", flush=True)
    elif args.replay:
        print(f"[train] WARNING replay pool {args.replay} missing; training without replay", flush=True)

    # two learning rates: pretrained weights (--lr) and the new head (--lr-head); no weight decay on 1-d parameters
    new_params = set(id(p) for p in head_model.new_parameters())
    groups = []
    for lr, sel in ((args.lr, lambda p: id(p) not in new_params), (args.lr_head, lambda p: id(p) in new_params)):
        dec = [p for p in head_model.parameters() if sel(p) and p.ndim >= 2]
        nodec = [p for p in head_model.parameters() if sel(p) and p.ndim < 2]
        if dec:
            groups.append({"params": dec, "weight_decay": args.weight_decay, "peak": lr})
        if nodec:
            groups.append({"params": nodec, "weight_decay": 0.0, "peak": lr})
    opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.98), eps=1e-6, fused=amp)

    # per-epoch renderings (option order and distractors drawn once per epoch)
    pool = distractor_pool(items, prng) if args.distract_frac > 0 else []

    def epoch_views(ep):
        views = []
        for it in items:
            perm = list(range(it.k))
            if ep > 0 and args.perm_aug != "none" and (args.perm_aug == "all" or it.qtype == "choice"):
                prng.shuffle(perm)
            dw = (pool, prng) if (pool and prng.random() < args.distract_frac) else None
            views.append((it, perm, *render(tok, it, perm, args.max_len, distract_with=dw)))
        return views

    t0 = time.time()
    all_epochs = []
    for ep in range(args.epochs):
        v = epoch_views(ep)
        lengths = np.array([len(x[2]) for x in v])
        all_epochs.append((v, make_batches(lengths, args.tokens_per_batch, rng)))
    total_steps = sum(len(b) for _, b in all_epochs)
    if replay is not None:
        total_steps = int(total_steps / (1 - args.replay_frac))
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup = int(args.warmup_frac * total_steps)
    print(f"[train] rendered {len(items)} x {args.epochs} epochs in {time.time() - t0:.0f}s; {total_steps} steps "
          f"(warmup {warmup}); mean len {np.mean([len(x[2]) for x in all_epochs[0][0]]):.0f}", flush=True)
    if replay is not None:
        rl = replay.lengths(min(len(replay), 2_000_000))
        replay_batches = make_batches(rl, args.tokens_per_batch, rng)
        rb_iter = iter(replay_batches)

    log = open(out / "train_log.jsonl", "a")
    (out / "train_args.json").write_text(json.dumps(vars(args), indent=1))
    step = 0
    run = defaultdict(float)
    run_n = 0
    done = False
    t0 = time.time()
    for ep, (views, batches) in enumerate(all_epochs):
        for idx in batches:
            if step >= total_steps:
                done = True
                break
            # replay steps: FLAN ATP batches through the MLM head
            while replay is not None and rng.random() < args.replay_frac and step < total_steps:
                try:
                    ridx = next(rb_iter)
                except StopIteration:
                    rb_iter = iter(replay_batches)
                    ridx = next(rb_iter)
                ids, attn, labels, _ = flan_collate(replay, ridx, rng, 0.8, 0.3)
                ids, attn, labels = ids.to(device), attn.to(device), labels.to(device)
                lr = lr_at(step, total_steps, warmup, 1.0)
                for g in opt.param_groups:
                    g["lr"] = lr * g["peak"]
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                    hidden = head_model.encoder(input_ids=ids, attention_mask=attn).last_hidden_state
                    pos = (labels != -100).nonzero()
                    vocab = head_model.vocab_logits_at(hidden, pos).float()
                    loss = torch.nn.functional.cross_entropy(vocab, labels[pos[:, 0], pos[:, 1]])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head_model.parameters(), args.clip)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                run["replay"] += float(loss)
                run["replay_n"] += 1
            if step >= total_steps:
                done = True
                break
            sel = [views[i] for i in idx]
            batch = {k: v.to(device, non_blocking=True) for k, v in collate(sel).items()}
            lr = lr_at(step, total_steps, warmup, 1.0)
            for g in opt.param_groups:
                g["lr"] = lr * g["peak"]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
                logits, valid = forward_options(head_model, batch)
            logits = logits.float()
            loss, parts = decision_loss(logits, valid, batch, losses)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(head_model.parameters(), args.clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            with torch.no_grad():
                g = batch["gold"]
                has = g >= 0
                acc = float((logits.argmax(-1)[has] == g[has]).float().mean()) if bool(has.any()) else 0.0
            run["loss"] += float(loss)
            run["acc"] += acc
            for k, v in parts.items():
                run[k] += v
            run_n += 1
            if step % args.log_every == 0 or step == total_steps:
                el = time.time() - t0
                rec = {"step": step, "epoch": ep, "loss": run["loss"] / run_n, "acc": run["acc"] / run_n,
                       **{k: run[k] / run_n for k in parts}, "lr": lr * opt.param_groups[0]["peak"], "grad_norm": float(gn),
                       "replay_loss": (run["replay"] / run["replay_n"]) if run["replay_n"] else None,
                       "peak_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1) if amp else None, "elapsed_s": el,
                       "eta_min": el / step * (total_steps - step) / 60, "batch": len(sel), "seq_len": int(batch["input_ids"].shape[1])}
                print("[train] " + json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
                run = defaultdict(float)
                run_n = 0
            if step % args.eval_every == 0 and step < total_steps:
                s = summarize(evaluate_cross(head_model, tok, dev_items, device, args.max_len, args.tokens_per_batch, amp))
                rec = {"step": step, "dev": {k: s[k] for k in ("n", "accuracy", "nll", "brier", "ece")}, "dev_by_type": s["by_type"]}
                print("[eval] " + json.dumps(rec), flush=True)
                log.write(json.dumps(rec) + "\n")
                log.flush()
        if done:
            break

    per_item = evaluate_cross(head_model, tok, dev_items, device, args.max_len, args.tokens_per_batch, amp)
    s = summarize(per_item)
    if args.calib_dev:
        calib_items = [it for it in (Item(r) for p in args.calib_dev for r in read_jsonl(p)) if it.gold is not None]
        temps = fit_temperatures(evaluate_cross(head_model, tok, calib_items, device, args.max_len, args.tokens_per_batch, amp))
        temps["_fitted_on"] = args.calib_dev
    else:
        temps = fit_temperatures(per_item)
    mlm.save_pretrained(out)
    tok.save_pretrained(out)
    head_model.save_head(out)
    (out / "calibration.json").write_text(json.dumps(temps, indent=1))
    final = {"steps": step, "elapsed_s": time.time() - t0, "n_train": len(items), "dev": s, "calibration": temps,
             "head": "h2", "loss": losses, "model": args.model}
    (out / "train_summary.json").write_text(json.dumps(final, indent=1))
    print("[train] done " + json.dumps({k: final[k] for k in ("steps", "elapsed_s")}) + " dev=" +
          json.dumps({k: round(s[k], 4) for k in ("accuracy", "nll", "brier", "ece")}), flush=True)
    print("[train] dev by source: " + json.dumps({k: round(v["accuracy"], 3) for k, v in s["by_source"].items()}), flush=True)


if __name__ == "__main__":
    main()
