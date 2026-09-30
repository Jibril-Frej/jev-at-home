"""Bi-encoder decision model (E): the state is encoded once, each option is scored against it by a small MLP.

  state   one pass of the encoder over the state text -> CLS vector s
  option  one query per option, the pair ("<type> question: <instructions>", "<id>: <description>") -> CLS vector q
  logit   MLP([s, q, s*q, |s-q|])  (LayerNorm, Linear, GELU, Linear(1))
Probabilities: softmax over the options of each question. The state encoding never depends on the question, so N
questions about one state cost one state pass (plus the short option queries).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

from .render import state_text

NEG = -1e4


class EmbedDecisionModel(nn.Module):
    def __init__(self, encoder, pooling: str = "cls"):
        super().__init__()
        assert pooling in ("cls", "mean")
        self.pooling = pooling
        self.encoder = encoder
        d = encoder.config.hidden_size
        self.scorer = nn.Sequential(nn.LayerNorm(4 * d), nn.Linear(4 * d, d), nn.GELU(), nn.Linear(d, 1))

    def new_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("encoder.")]

    def encode(self, ids, att):
        h = self.encoder(input_ids=ids, attention_mask=att).last_hidden_state
        if self.pooling == "mean":
            m = att.unsqueeze(-1).to(h.dtype)
            return (h * m).sum(1) / m.sum(1).clamp(min=1)
        return h[:, 0]

    def forward(self, b):
        """b: build_batch output. Returns (logits [B, kmax], valid [B, kmax])."""
        s = self.encode(b["state_ids"], b["state_att"])[b["state_index"]]
        q = self.encode(b["query_ids"], b["query_att"])
        score = self.scorer(torch.cat([s, q, s * q, (s - q).abs()], -1)).squeeze(-1)
        logits = torch.full((b["n_rows"], b["kmax"]), NEG, device=score.device, dtype=torch.float32)
        logits[b["row_index"], b["option_slot"]] = score.float()
        return logits, logits > NEG / 2

    def save(self, out):
        out = Path(out)
        self.encoder.save_pretrained(out)
        torch.save({k: v for k, v in self.state_dict().items() if not k.startswith("encoder.")}, out / "head.pt")
        (out / "decision_config.json").write_text(json.dumps({"head": "emb", "variant": "bi", "cross_layers": 2, "evidence": False,
                                                                "pooling": self.pooling}))

    @classmethod
    def load(cls, path, device="cpu", dtype=torch.float32):
        from transformers import AutoModel
        path = Path(path)
        cfg = json.loads((path / "decision_config.json").read_text())
        enc = AutoModel.from_pretrained(path, dtype=dtype, attn_implementation="sdpa")
        m = cls(enc, pooling=cfg.get("pooling", "cls"))
        missing, unexpected = m.load_state_dict(torch.load(path / "head.pt", map_location="cpu"), strict=False)
        missing = [x for x in missing if not x.startswith("encoder.")]
        assert not missing and not unexpected, (missing, unexpected)
        return m.to(device=device, dtype=dtype).eval()


# ---------------------------------------------------------------- text + batching (shared by training and evaluation)
def texts_for(req: dict, qtype: str):
    """Canonical request -> (state text, instruction text, option texts)."""
    return state_text(req["state"]), f"{qtype} question: {req['question']}", [o["description"] for o in req["options"]]


def build_batch(tok, rows, device, max_state=512, max_query=160):
    """rows: list of (state_text, instr_text, [option texts]). Identical states are encoded once."""
    uniq, s_index = {}, []
    for st, _, _ in rows:
        s_index.append(uniq.setdefault(st, len(uniq)))
    S = tok(list(uniq), padding=True, truncation=True, max_length=max_state, return_tensors="pt")
    state_index, row_index, slot = [], [], []
    for r, (_, _, crits) in enumerate(rows):
        for j in range(len(crits)):
            state_index.append(s_index[r]); row_index.append(r); slot.append(j)
    Q = tok([instr for _, instr, cs in rows for _ in cs], [c for _, _, cs in rows for c in cs], padding=True,
            truncation="longest_first", max_length=max_query, return_tensors="pt")
    b = {"state_ids": S["input_ids"], "state_att": S["attention_mask"], "query_ids": Q["input_ids"], "query_att": Q["attention_mask"],
         "state_index": torch.tensor(state_index), "row_index": torch.tensor(row_index), "option_slot": torch.tensor(slot),
         "n_rows": len(rows), "kmax": max(len(c) for _, _, c in rows)}
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}


def model_pooling(model_id: str) -> str:
    """The sentence-embedding pooling a checkpoint was trained with (sentence-transformers 1_Pooling config); cls if unknown."""
    try:
        p = Path(model_id) / "1_Pooling" / "config.json"
        if not p.exists():
            from huggingface_hub import hf_hub_download
            p = Path(hf_hub_download(model_id, "1_Pooling/config.json"))
        return "mean" if json.loads(p.read_text()).get("pooling_mode_mean_tokens") else "cls"
    except Exception:
        return "cls"
