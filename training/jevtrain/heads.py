"""Decision head of the cross-encoders (Ettin-1B, L, B) on top of a ModernBERT-style masked LM.

MarkerHead: the encoder's hidden state at each option's [MASK] marker, plus a learned embedding of the question type,
goes through a small scorer (LayerNorm, Linear, GELU, Linear(1)) -> one logit per option; softmax over the options
gives the probabilities. The MLM head / decoder are not used for decisions; they are kept for the FLAN replay batches
(a regulariser during training) and dropped at export.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn

QTYPES = {"choice": 0, "score": 1, "noul": 2}
NEG = -1e4


class MarkerHead(nn.Module):
    def __init__(self, mlm):
        super().__init__()
        self.mlm = mlm
        d = mlm.config.hidden_size
        self.type_emb = nn.Embedding(3, d)
        nn.init.normal_(self.type_emb.weight, std=0.02)
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    @property
    def encoder(self):
        return self.mlm.model

    def new_parameters(self):
        return list(self.type_emb.parameters()) + list(self.scorer.parameters())

    def vocab_logits_at(self, hidden, positions):
        """hidden [B, L, d], positions [N, 2] (batch idx, pos) -> [N, vocab] (FLAN replay only)."""
        h = hidden[positions[:, 0], positions[:, 1]]
        return self.mlm.decoder(self.mlm.head(h))

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = h + self.type_emb(qtype)[:, None, :]
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        logits = self.scorer(torch.gather(h, 1, idx)).squeeze(-1).float()
        return logits.masked_fill(~marker_mask, NEG), marker_mask

    def save_head(self, out: Path):
        sd = {k: v for k, v in self.state_dict().items() if not k.startswith("mlm.")}
        torch.save(sd, Path(out) / "head.pt")
        (Path(out) / "decision_config.json").write_text(json.dumps({"head": "h2", "head_layers": 0, "typed": False, "ordinal": False,
                                                                     "order_invariant": False, "evidence": False}))

    def load_head(self, out: Path):
        sd = torch.load(Path(out) / "head.pt", map_location="cpu")
        missing, unexpected = self.load_state_dict(sd, strict=False)
        missing = [m for m in missing if not m.startswith("mlm.")]
        assert not missing and not unexpected, (missing, unexpected)


def load_cross_encoder(path, device="cpu", dtype=torch.float32):
    """A trained cross-encoder checkpoint (model.safetensors + head.pt) -> MarkerHead in eval mode."""
    from transformers import AutoModelForMaskedLM
    mlm = AutoModelForMaskedLM.from_pretrained(path, dtype=dtype, attn_implementation="sdpa")
    mlm.config.sparse_prediction = False
    mlm.sparse_prediction = False
    head = MarkerHead(mlm)
    head.load_head(path)
    return head.to(device=device, dtype=dtype).eval()
