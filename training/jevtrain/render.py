"""Canonical options and the cross-encoder token layout (the same as jevhome's Rust preprocessing).

Canonical options: noul = [true, false] (descriptions from criteria, else "The proposition is true/false."),
choice = criteria order, score = levels 0..k-1; every description is prefixed with "<id>: ".

Cross-encoder layout (one [MASK] marker before each option; the head reads the hidden state at each marker):
    [CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]
"""
from __future__ import annotations

import json

MASK_ID, PAD_ID, CLS_ID, SEP_ID = 50284, 50283, 50281, 50282  # ModernBERT / Ettin vocabulary


def canonical_request(row: dict, option_order=None) -> dict:
    """{'id', 'state', 'question', 'options': [{id, description}]} in canonical (or permuted) order."""
    q = row["question"]
    qtype, crit = q["type"], q.get("criteria")
    if qtype == "noul":
        options = [{"id": k, "description": (crit or {}).get(k, f"The proposition is {k}.")} for k in ("true", "false")]
    elif qtype == "choice":
        options = [{"id": k, "description": v or k} for k, v in crit.items()]
    else:
        options = [{"id": str(i), "description": lvl} for i, lvl in enumerate(crit)]
    for o in options:
        o["description"] = o["id"] + ": " + o["description"]
    if option_order is not None:
        options = [options[i] for i in option_order]
    return {"id": row["id"], "state": row["state"], "question": q["instructions"], "options": options}


def option_ids(row: dict) -> list:
    return [o["id"] for o in canonical_request(row)["options"]]


def state_text(state) -> str:
    """String form of a state: str as-is, anything else via json.dumps(ensure_ascii=False)."""
    return state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)


def h2_ids(tok, req: dict, qtype: str, max_len: int = 512, head_max_len: int = 320, opt_max: int = 48):
    """Returns (ids, marker_positions). Options are cut to opt_max tokens, question + options to head_max_len,
    and the state fills what is left of max_len (cut from its end)."""
    mask_tok = tok.mask_token
    ins = str(req["question"]).replace(mask_tok, " ")
    head = tok(f"{qtype} question: {ins}", add_special_tokens=False)["input_ids"]
    opts = []
    for o in req["options"]:
        t = tok(" " + o["description"].replace(mask_tok, " "), add_special_tokens=False,
                truncation=True, max_length=opt_max)["input_ids"]
        opts.append([MASK_ID] + t)
    budget = head_max_len - sum(len(o) for o in opts)
    if budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opts)))
        opts = [o[:per] for o in opts]
        budget = head_max_len - sum(len(o) for o in opts)
    head = head[: max(8, budget)]
    ids = [CLS_ID] + head + [SEP_ID]
    markers = []
    for o in opts:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(SEP_ID)
    st = tok(state_text(req["state"]).replace(mask_tok, " "), add_special_tokens=False)["input_ids"]
    room = max(0, max_len - len(ids) - 1)
    ids = (ids + st[:room] + [SEP_ID])[:max_len]
    return ids, markers
