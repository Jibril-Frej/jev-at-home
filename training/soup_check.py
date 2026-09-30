"""Safety check for a weight soup: accuracy on the seeds' dev file vs the best seed.
If the soup is more than --tol below the best seed, replace it by a copy of the best seed (soup_of.txt records it).
usage: soup_check.py --soup checkpoints/SOUP-X --seeds checkpoints/A checkpoints/B ..."""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from jevtrain.decision import Item, predict
from jevtrain.records import read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--soup", required=True)
    ap.add_argument("--seeds", nargs="+", required=True)
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    seed_acc = {s: json.load(open(Path(s) / "train_summary.json"))["dev"]["accuracy"] for s in a.seeds}
    dev = json.load(open(Path(a.seeds[0]) / "train_args.json"))["dev"]
    items = [it for it in (Item(r) for p in dev for r in read_jsonl(p)) if it.gold is not None]
    per_item = predict(a.soup, items, device=a.device)
    soup_acc = sum(int(np.argmax(p)) == it.gold for it, _, p in per_item) / len(per_item)
    best = max(seed_acc, key=seed_acc.get)
    print(f"[soup_check] soup dev acc {soup_acc:.4f}; seeds {json.dumps({Path(k).name: round(v, 4) for k, v in seed_acc.items()})}")
    if soup_acc < seed_acc[best] - a.tol:
        shutil.rmtree(a.soup)
        shutil.copytree(best, a.soup)
        (Path(a.soup) / "soup_of.txt").write_text(f"FALLBACK: soup dev {soup_acc:.4f} < best seed {seed_acc[best]:.4f}; copied {best}\n")
        print(f"[soup_check] soup rejected -> using best seed {best}")


if __name__ == "__main__":
    main()
