"""Instruction-tune a ModernBERT masked LM with Answer Token Prediction (Clavié et al. 2025, arXiv 2502.03793).

Objective per training example (section 2.4 of the paper, "final objective choice"):
  * with probability --atp-frac (0.8): Answer Token Prediction — the single answer token after the
    "[unused0]" anchor is replaced by [MASK] and is the only labelled position;
  * otherwise: "dummy MLM" — 30 % of the non-special tokens are replaced by [MASK] and every masked
    position is labelled with the [MASK] token id itself (the label-dropout bug the authors kept).
Loss is cross-entropy over the labelled positions only (sparse prediction through the MLM head).

Data: the flat token stream written by prepare_flan.py; --n-examples takes the first n examples of the
shuffled pool, so the 2M / 5M / 20M runs are nested subsets of one another.

usage (B's backbone): train_atp.py --data data/flan_atp_ny --n-examples 20000000 --out checkpoints/modernbert-base-atp-20m-ny
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from jevtrain.flan import CLS_ID, MASK_ID, PAD_ID, SEP_ID, Pool, collate, lr_at, make_batches


@torch.no_grad()
def evaluate(model, pool: Pool, device, tokens_per_batch, max_batches=None):
    model.eval()
    rng = np.random.default_rng(0)
    lengths = pool.lengths(len(pool))
    batches = make_batches(lengths, tokens_per_batch, rng)
    if max_batches:
        batches = batches[:max_batches]
    n = correct = 0
    loss_sum = 0.0
    for idx in batches:
        ids, attn, labels, _ = collate(pool, idx, rng, 1.0, 0.0, eval_mode=True)
        ids, attn, labels = ids.to(device), attn.to(device), labels.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            out = model(input_ids=ids, attention_mask=attn, labels=labels)
        tgt = labels[labels != -100]
        pred = out.logits.float().argmax(-1)
        correct += int((pred == tgt).sum())
        n += int(tgt.numel())
        loss_sum += float(out.loss) * int(tgt.numel())
    model.train()
    return {"eval_loss": loss_sum / max(1, n), "eval_atp_acc": correct / max(1, n), "eval_n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/flan_atp_ny")
    ap.add_argument("--n-examples", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="answerdotai/ModernBERT-base")
    ap.add_argument("--tokens-per-batch", type=int, default=65536)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--betas", default="0.9,0.98")
    ap.add_argument("--eps", type=float, default=1e-6)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--atp-frac", type=float, default=0.8)
    ap.add_argument("--mlm-prob", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--attn", default="sdpa")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--eval-every", type=int, default=2000)
    ap.add_argument("--eval-batches", type=int, default=40)
    ap.add_argument("--save-every", type=int, default=5000)
    ap.add_argument("--max-steps", type=int, default=None, help="debug: stop early")
    ap.add_argument("--device", default=None)
    ap.add_argument("--grad-accum", type=int, default=1, help="split each batch into this many micro-batches (same effective batch, less memory)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    from transformers import AutoModelForMaskedLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    assert tok.mask_token_id == MASK_ID and tok.pad_token_id == PAD_ID and tok.cls_token_id == CLS_ID and tok.sep_token_id == SEP_ID
    model = AutoModelForMaskedLM.from_pretrained(args.model, dtype=torch.float32, attn_implementation=args.attn)
    model.config.sparse_prediction = True
    model.sparse_prediction = True
    model.to(device).train()

    train = Pool(str(Path(args.data) / "train_pool"))
    held = Pool(str(Path(args.data) / "heldout"))
    n = min(args.n_examples, len(train))
    lengths = train.lengths(n)
    print(f"[train] {n:,} examples, {int(lengths.sum()):,} tokens, mean len {lengths.mean():.0f}, "
          f"model {args.model} {sum(p.numel() for p in model.parameters())/1e6:.0f}M params, device {device}", flush=True)

    epochs_batches = [make_batches(lengths, args.tokens_per_batch, rng) for _ in range(args.epochs)]
    total_steps = sum(len(b) for b in epochs_batches)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup = int(args.warmup_frac * total_steps)
    print(f"[train] {total_steps} steps ({len(epochs_batches[0])}/epoch), warmup {warmup}, peak lr {args.lr}, "
          f"tokens/batch {args.tokens_per_batch}", flush=True)

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        (no_decay if p.ndim < 2 else decay).append(p)
    b1, b2 = (float(x) for x in args.betas.split(","))
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr, betas=(b1, b2), eps=args.eps, fused=device.type == "cuda")

    log = open(out / "train_log.jsonl", "a")
    (out / "train_args.json").write_text(json.dumps(vars(args), indent=1))
    step = 0
    t0 = time.time()
    tok_seen = 0
    run_loss = run_atp_correct = run_atp_n = run_examples = 0.0
    done = False
    for ep, batches in enumerate(epochs_batches):
        for idx in batches:
            if step >= total_steps:
                done = True; break
            lr = lr_at(step, total_steps, warmup, args.lr)
            for g in opt.param_groups:
                g["lr"] = lr
            micro = [m for m in np.array_split(np.asarray(idx), min(args.grad_accum, len(idx))) if len(m)]
            parts = [collate(train, m.tolist(), rng, args.atp_frac, args.mlm_prob) for m in micro]
            n_lab = sum(int((p[2] != -100).sum()) for p in parts)
            loss = 0.0
            for ids, attn, labels, is_atp in parts:
                ids, attn, labels = ids.to(device, non_blocking=True), attn.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    outp = model(input_ids=ids, attention_mask=attn, labels=labels)
                # weight each micro-batch by its share of labelled tokens so the sum equals the full-batch mean loss
                part = outp.loss * (int((labels != -100).sum()) / max(1, n_lab))
                part.backward()
                loss = loss + part.detach()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            opt.step(); opt.zero_grad(set_to_none=True)
            step += 1
            tok_seen += sum(int(p[1].sum()) for p in parts)
            run_loss += loss.detach().item(); run_examples += 1
            # ATP accuracy on this batch's ATP rows (labels at the answer slot that are not the dummy mask label)
            with torch.no_grad():
                lab = labels[labels != -100]
                real = lab != MASK_ID
                if real.any():
                    pred = outp.logits.float().argmax(-1)
                    run_atp_correct += int((pred[real] == lab[real]).sum()); run_atp_n += int(real.sum())
            if step % args.log_every == 0 or step == total_steps:
                el = time.time() - t0
                rec = {"step": step, "epoch": ep, "loss": run_loss / run_examples, "atp_acc": run_atp_correct / max(1, run_atp_n),
                       "lr": lr, "grad_norm": float(gn), "tokens_seen": tok_seen, "tok_per_s": tok_seen / el,
                       "elapsed_s": el, "eta_min": el / step * (total_steps - step) / 60, "batch": len(idx), "seq_len": ids.shape[1]}
                print("[train] " + json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
                log.write(json.dumps(rec) + "\n"); log.flush()
                run_loss = run_atp_correct = run_atp_n = run_examples = 0.0
            if step % args.eval_every == 0 or step == total_steps:
                ev = evaluate(model, held, device, args.tokens_per_batch, args.eval_batches)
                ev["step"] = step
                print("[eval] " + json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in ev.items()}), flush=True)
                log.write(json.dumps(ev) + "\n"); log.flush()
            if step % args.save_every == 0 and step < total_steps:
                model.save_pretrained(out / "last"); tok.save_pretrained(out / "last")
                (out / "last" / "step.json").write_text(json.dumps({"step": step}))
        if done:
            break
    model.config.sparse_prediction = False  # inference default; the checkpoint returns full logits
    model.save_pretrained(out); tok.save_pretrained(out)
    final = {"steps": step, "elapsed_s": time.time() - t0, "tokens_seen": tok_seen, "n_examples": n,
             "final_eval": evaluate(model, held, device, args.tokens_per_batch, None)}
    (out / "train_summary.json").write_text(json.dumps(final, indent=1))
    print("[train] done " + json.dumps(final), flush=True)


if __name__ == "__main__":
    main()
