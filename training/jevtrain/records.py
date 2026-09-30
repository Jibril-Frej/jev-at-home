"""Decision records: one JSON object per line (written by from_hf.py).

  id, family, source, contrast_of
  state         str, or a JSON value (object) when the dataset's state_format is json
  question      {"type": "noul" | "choice" | "score", "instructions": str, "criteria": dict | list | None}
  expected      gold answer: "yes" / "no" for noul, an option id otherwise; None when there is none
  teacher       {"model", "option_ids", "probs_avg"} or None; probabilities in the canonical option order
                of jevtrain.render.canonical_request (noul = [true, false])
"""
from __future__ import annotations

import json


def read_jsonl(path) -> list:
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows, path) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for d in rows:
            fh.write(json.dumps(d, ensure_ascii=False) + "\n")


def gold_index(row: dict, option_ids: list) -> int | None:
    """Index of the gold option in canonical order, or None when there is no gold answer."""
    exp = row.get("expected")
    if exp is None:
        return None
    if row["question"]["type"] == "noul":
        return option_ids.index("true" if exp == "yes" else "false")
    return option_ids.index(str(exp))
