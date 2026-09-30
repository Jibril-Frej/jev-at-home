"""Hugging Face dataset jevhome/jevhome-decisions (parquet, one flat row per decision) -> training records (jsonl).

usage: from_hf.py [--src jevhome/jevhome-decisions | <local dir with data/*.parquet>] [--out data/jevhome]
Writes <out>/{train,dev,calibration}.jsonl in the format of jevtrain/records.py.
"""
import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq

from jevtrain.records import write_jsonl
from jevtrain.render import option_ids

SPLITS = ("train", "dev", "calibration")


def record(r: dict) -> dict:
    qtype = r["question_type"]
    row = {"id": r["id"], "family": r["family"], "source": r["source"], "contrast_of": r["contrast_of"],
           "state": json.loads(r["state"]) if r["state_format"] == "json" else r["state"],
           "question": {"type": qtype, "instructions": r["instructions"],
                        "criteria": None if r["criteria"] is None else json.loads(r["criteria"])}}
    g = r["gold"]
    row["expected"] = None if g is None else ({"true": "yes", "false": "no"}[g] if qtype == "noul" else g)
    oids = [o["id"] for o in r["options"]]
    assert option_ids(row) == oids, r["id"]  # the dataset's option order is the canonical one
    row["teacher"] = None if r["teacher_probs"] is None else {"model": r["teacher"], "option_ids": oids,
                                                              "probs_avg": [float(p) for p in r["teacher_probs"]]}
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="jevhome/jevhome-decisions", help="HF dataset id or a local copy")
    ap.add_argument("--out", default="data/jevhome")
    a = ap.parse_args()
    src = Path(a.src)
    if not (src / "data").is_dir():
        from huggingface_hub import snapshot_download
        src = Path(snapshot_download(a.src, repo_type="dataset", allow_patterns=["data/*.parquet"]))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for sp in SPLITS:
        rows = [record(r) for r in pq.read_table(src / "data" / f"{sp}.parquet").to_pylist()]
        write_jsonl(rows, out / f"{sp}.jsonl")
        print(f"{sp}: {len(rows):,} rows -> {out / f'{sp}.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
