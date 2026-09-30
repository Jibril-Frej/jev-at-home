"""Uniform weight averaging ("model soup") of checkpoints trained with the same recipe from the same start.

usage: soup.py --out checkpoints/<name> checkpoints/<run-s0> checkpoints/<run-s1> [...]
Averages model.safetensors (encoder + MLM head) and head.pt (marker head) key by key; copies config, tokenizer and
decision_config from the first checkpoint; writes no calibration.json (fit one afterwards if needed).
"""
import argparse
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("ckpts", nargs="+")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    first = Path(args.ckpts[0])
    for f in first.iterdir():
        if f.name not in ("model.safetensors", "head.pt", "calibration.json", "calibration_u.json", "train_log.jsonl", "last") and f.is_file():
            shutil.copy(f, out / f.name)
    sds = [load_file(str(Path(c) / "model.safetensors")) for c in args.ckpts]
    avg = {k: sum(sd[k].float() for sd in sds) / len(sds) for k in sds[0]}
    avg = {k: v.to(sds[0][k].dtype) for k, v in avg.items()}
    save_file(avg, str(out / "model.safetensors"), metadata={"format": "pt"})
    if (first / "head.pt").exists():
        hs = [torch.load(Path(c) / "head.pt", map_location="cpu") for c in args.ckpts]
        torch.save({k: sum(h[k].float() for h in hs) / len(hs) for k in hs[0]}, out / "head.pt")
    (out / "soup_of.txt").write_text("\n".join(args.ckpts) + "\n")
    print(f"soup of {len(args.ckpts)} checkpoints -> {out}")


if __name__ == "__main__":
    main()
