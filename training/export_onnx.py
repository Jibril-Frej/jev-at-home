"""Export a trained checkpoint (cross-encoder or bi-encoder) to the fp32 ONNX layout jevhome loads.

usage: export_onnx.py checkpoints/<soup> models/<name>      (the checkpoint needs calibration_u.json: fit_temperature.py)

Cross-encoder (Ettin-1B, L, B) -> model.onnx
    inputs input_ids [n, len], attention_mask [n, len], marker_pos [n, k] (position of each option's [MASK]), qtype [n]
    output logits [n, k]
Bi-encoder (E) -> encoder.onnx (ids, att -> CLS vector vec) + scorer.onnx (state vector s, option vector q -> logit),
    so jevhome can encode a state once and reuse it for every question about it.
Both: the last encoder layer is computed only at the positions that are read (the option markers / the CLS token);
every position still serves as key/value there, so the answers are those of the PyTorch model. The MLM head is dropped.
Graphs over 2 GB are saved as <graph>.onnx + one <graph>.onnx.data file. Copied alongside: tokenizer.json,
tokenizer_config.json, config.json, calibration_u.json, decision_config.json.
"""
import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import onnx
import torch
import torch.nn.functional as F

from jevtrain.embed_model import EmbedDecisionModel, build_batch, texts_for
from jevtrain.heads import QTYPES, load_cross_encoder
from jevtrain.render import PAD_ID, canonical_request, h2_ids

# two requests to trace the graphs with (batch 2, different lengths and option counts)
EXAMPLES = [
    {"id": "ex0", "state": "The user asked to cancel their subscription and get a refund for last month.",
     "question": {"type": "choice", "instructions": "Which team should handle this request?",
                  "criteria": {"billing": "Payments, refunds and invoices", "support": "Product problems", "sales": "New purchases"}}},
    {"id": "ex1", "state": {"temperature_c": 31, "humidity": 0.4},
     "question": {"type": "noul", "instructions": "Is it hot outside?", "criteria": None}},
]
FILES = ("tokenizer.json", "tokenizer_config.json", "config.json", "calibration_u.json", "decision_config.json")


class SlimEncoder(torch.nn.Module):
    """ModernBERT encoder whose last layer is computed only at `pos` [B, P]; returns final-normed hidden [B, P, d]."""
    def __init__(self, enc):
        super().__init__()
        self.enc = enc

    def forward(self, ids, att, pos):
        from transformers.models.modernbert.modeling_modernbert import rotate_half
        e, cfg = self.enc, self.enc.config
        B, L = ids.shape
        h = e.embeddings(input_ids=ids)
        ar = torch.arange(L, device=ids.device)
        key = att.bool()[:, None, None, :]                                        # [B, 1, 1, L]
        win = (ar[:, None] - ar[None, :]).abs() <= cfg.sliding_window              # [L, L], inclusive as in HF
        masks = {"full_attention": key.expand(B, 1, L, L), "sliding_attention": key & win[None, None]}
        pid = ar[None]
        rope = {t: e.rotary_emb(h, pid, t) for t in set(cfg.layer_types)}
        for layer in e.layers[:-1]:
            h = layer(h, attention_mask=masks[layer.attention_type], position_embeddings=rope[layer.attention_type])
        last, a = e.layers[-1], e.layers[-1].attn
        d, nh, hd = h.shape[-1], cfg.num_attention_heads, a.head_dim
        x = last.attn_norm(h)
        gidx = pos[:, :, None].expand(-1, -1, d)
        W, b = a.Wqkv.weight, a.Wqkv.bias
        q = F.linear(torch.gather(x, 1, gidx), W[:d], None if b is None else b[:d])          # [B, P, d]
        kv = F.linear(x, W[d:], None if b is None else b[d:]).view(B, L, 2, nh, hd)
        k, v = kv[:, :, 0].transpose(1, 2), kv[:, :, 1].transpose(1, 2)                      # [B, nh, L, hd]
        q = q.view(B, -1, nh, hd).transpose(1, 2)                                           # [B, nh, P, hd]
        cos, sin = rope[last.attention_type]                                                # [1, L, hd]
        cq, sq = cos[0][pos][:, None], sin[0][pos][:, None]
        q = q * cq + rotate_half(q) * sq
        k = k * cos[:, None] + rotate_half(k) * sin[:, None]
        s = torch.matmul(q, k.transpose(2, 3)) * hd ** -0.5                                 # [B, nh, P, L]
        m = key
        if last.attention_type == "sliding_attention":
            m = m & ((pos[:, :, None] - ar[None, None, :]).abs() <= cfg.sliding_window)[:, None]
        s = s.masked_fill(~m, torch.finfo(s.dtype).min)
        o = torch.matmul(torch.softmax(s, -1), v).transpose(1, 2).reshape(B, -1, d)
        hq = torch.gather(h, 1, gidx) + a.Wo(o)
        hq = hq + last.mlp(last.mlp_norm(hq))
        return e.final_norm(hq)


class CrossEncoder(torch.nn.Module):
    def __init__(self, head):
        super().__init__()
        self.enc, self.head = SlimEncoder(head.encoder), head

    def forward(self, input_ids, attention_mask, marker_pos, qtype):
        m = self.enc(input_ids, attention_mask, marker_pos) + self.head.type_emb(qtype)[:, None, :]
        return self.head.scorer(m).squeeze(-1).float()


class BiEncoder(torch.nn.Module):
    def __init__(self, emb):
        super().__init__()
        self.enc = SlimEncoder(emb.encoder)

    def forward(self, ids, att):
        return self.enc(ids, att, torch.zeros_like(ids[:, :1]))[:, 0]


class Scorer(torch.nn.Module):
    def __init__(self, emb):
        super().__init__()
        self.scorer = emb.scorer

    def forward(self, s, q):
        return self.scorer(torch.cat([s, q, s * q, (s - q).abs()], -1)).squeeze(-1)


def cross_feeds(tok):
    reqs = [canonical_request(r) for r in EXAMPLES]
    enc = [h2_ids(tok, q, r["question"]["type"]) for q, r in zip(reqs, EXAMPLES)]
    L, K = max(len(e[0]) for e in enc), max(len(e[1]) for e in enc)
    ids = np.full((len(enc), L), PAD_ID, np.int64)
    att = np.zeros((len(enc), L), np.int64)
    mpos = np.zeros((len(enc), K), np.int64)
    for i, (x, mk) in enumerate(enc):
        ids[i, :len(x)] = x; att[i, :len(x)] = 1; mpos[i, :len(mk)] = mk
    qt = np.array([QTYPES[r["question"]["type"]] for r in EXAMPLES], np.int64)
    return [torch.from_numpy(a) for a in (ids, att, mpos, qt)]


def consolidate(src: Path, dst: Path):
    """Copy the exported graphs; a graph with external tensors is re-saved as <g>.onnx + a single <g>.onnx.data."""
    for g in ("model", "encoder", "scorer"):
        f = src / f"{g}.onnx"
        if not f.exists():
            continue
        if f.stat().st_size > 2**31 - 1 or any(t.data_location == onnx.TensorProto.EXTERNAL
                                                for t in onnx.load(f, load_external_data=False).graph.initializer):
            m = onnx.load(f)  # loads every external tensor, wherever it is stored
            onnx.save_model(m, dst / f"{g}.onnx", save_as_external_data=True, all_tensors_to_one_file=True,
                            location=f"{g}.onnx.data", size_threshold=1024)
        else:
            shutil.copy2(f, dst / f.name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("out")
    args = ap.parse_args()
    from transformers import AutoTokenizer
    path, out = Path(args.ckpt), Path(args.out)
    if not (path / "calibration_u.json").exists():
        raise SystemExit(f"{path}/calibration_u.json missing: run fit_temperature.py --model {path} first")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(path)
    kind = json.loads((path / "decision_config.json").read_text())["head"]
    with tempfile.TemporaryDirectory(dir=out) as tmp, torch.inference_mode():
        tmp = Path(tmp)
        if kind == "h2":
            model = CrossEncoder(load_cross_encoder(path)).eval()
            names = ["input_ids", "attention_mask", "marker_pos", "qtype"]
            dyn = {"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                   "marker_pos": {0: "batch", 1: "k"}, "qtype": {0: "batch"}}
            torch.onnx.export(model, tuple(cross_feeds(tok)), str(tmp / "model.onnx"), input_names=names, output_names=["logits"],
                              dynamic_axes=dyn, opset_version=18, dynamo=False, external_data=True)
        else:
            emb = EmbedDecisionModel.load(path, "cpu", torch.float32)
            assert emb.pooling == "cls", "the exported encoder reads the CLS vector"
            enc, scorer = BiEncoder(emb).eval(), Scorer(emb).eval()
            b = build_batch(tok, [texts_for(canonical_request(r), r["question"]["type"]) for r in EXAMPLES], "cpu")
            ex = (b["query_ids"], b["query_att"])
            torch.onnx.export(enc, ex, str(tmp / "encoder.onnx"), input_names=["ids", "att"], output_names=["vec"],
                              dynamic_axes={"ids": {0: "n", 1: "len"}, "att": {0: "n", 1: "len"}}, opset_version=18,
                              dynamo=False, external_data=True)
            v = enc(*ex)
            torch.onnx.export(scorer, (v, v), str(tmp / "scorer.onnx"), input_names=["s", "q"], output_names=["logits"],
                              dynamic_axes={"s": {0: "n"}, "q": {0: "n"}}, opset_version=18, dynamo=False)
        consolidate(tmp, out)
    for f in FILES:
        shutil.copy2(path / f, out / f)
    print(f"exported {path} ({kind}) in {time.time() - t0:.0f}s -> {out}: {sorted(p.name for p in out.iterdir())}", flush=True)


if __name__ == "__main__":
    main()
