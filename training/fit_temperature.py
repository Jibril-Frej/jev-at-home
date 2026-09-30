"""Fit per-(type, K bucket) temperatures for a finished checkpoint on any row file (no retraining), fp32 inference.
The result (calibration_u.json) ships with the model; jevhome divides the option logits by T before the softmax.
usage: fit_temperature.py --model checkpoints/<soup> --dev data/jevhome/calibration.jsonl --t-min 0.05"""
import argparse
import json
from pathlib import Path

from jevtrain.decision import Item, fit_temperatures, predict
from jevtrain.records import read_jsonl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dev", nargs="+", required=True)
    ap.add_argument("--out", default=None, help="default: <model>/calibration_u.json")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--t-min", type=float, default=0.5, help="lower bound; soups need sharpening (e.g. 0.05)")
    args = ap.parse_args()
    items = [it for it in (Item(r) for p in args.dev for r in read_jsonl(p)) if it.gold is not None]
    temps = fit_temperatures(predict(args.model, items, device=args.device), t_min=args.t_min)
    temps["_fitted_on"] = args.dev
    out = Path(args.out or (Path(args.model) / "calibration_u.json"))
    out.write_text(json.dumps(temps, indent=1))
    print(json.dumps({k: v["T"] for k, v in temps.items() if not k.startswith("_")}))


if __name__ == "__main__":
    main()
