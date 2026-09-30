"""Shared by the training and evaluation scripts: training items, distractor augmentation, batching, evaluation,
metrics and temperature fitting, for both model kinds (cross-encoder "h2" and bi-encoder "emb")."""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .embed_model import EmbedDecisionModel, build_batch, texts_for
from .heads import QTYPES, load_cross_encoder
from .losses import ce_loss, kl_loss, masked_log_softmax
from .records import gold_index
from .render import PAD_ID, canonical_request, h2_ids


def bucket(qtype: str, k: int) -> str:
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return f"{qtype}:{size}"


class Item:
    """One decision: canonical request, gold index and training target (teacher probabilities, else one-hot gold)."""
    __slots__ = ("row", "req", "k", "qtype", "gold", "target", "source")

    def __init__(self, row):
        self.row = row
        self.req = canonical_request(row)
        self.k = len(self.req["options"])
        self.qtype = row["question"]["type"]
        oids = [o["id"] for o in self.req["options"]]
        g = gold_index(row, oids)
        t = row.get("teacher")
        tp = None
        if t and t.get("probs_avg"):
            assert t["option_ids"] == oids, row["id"]
            tp = list(t["probs_avg"])
        self.gold = g
        self.target = tp if tp is not None else ([0.0] * self.k if g is None else [1.0 if i == g else 0.0 for i in range(self.k)])
        self.source = row.get("source", "?")


# ---------------------------------------------------------------- distractor augmentation
DISTRACT_TEMPLATES = ["{s}\n\nNote: {d}", "{s}\n\nUnrelated: {d}", "{s} {d}", "{d}\n\n{s}", "{s}\n({d})", "Context: {d}\n\n{s}"]


def distractor_pool(items, prng, max_words=28):
    """First sentences of the text states (4 to max_words words), tagged with their source."""
    pool = []
    for it in items:
        st = it.row["state"]
        if isinstance(st, str):
            first = st.replace("\n", " ").split(". ")[0].strip()
            w = first.split()
            if 4 <= len(w) <= max_words:
                pool.append((it.source, first.rstrip(".") + "."))
    prng.shuffle(pool)
    return pool


def distract(req, item, pool, prng):
    """Append/prepend an unrelated sentence from another source; the gold label cannot change."""
    if not pool:
        return req
    for _ in range(5):
        src, d = pool[prng.randrange(len(pool))]
        if src != item.source:
            break
    req = dict(req)
    st = req["state"]
    if isinstance(st, str):
        req["state"] = prng.choice(DISTRACT_TEMPLATES).format(s=st, d=d)
    elif isinstance(st, dict):
        st = dict(st)
        st[prng.choice(["note", "unrelated", "misc", "aside"])] = d
        req["state"] = st
    return req


# ---------------------------------------------------------------- cross-encoder batches
def render(tok, item: Item, perm, max_len: int, distract_with=None):
    """(ids, marker positions) of one view of an item: options in `perm` order, optionally with a distractor."""
    req = dict(item.req)
    if distract_with is not None:
        req = distract(req, item, distract_with[0], distract_with[1])
    req["options"] = [item.req["options"][i] for i in perm]
    return h2_ids(tok, req, item.qtype, max_len)


def collate(views):
    """views: list of (item, perm, ids, markers). Returns tensors (option logits come out in perm order)."""
    B = len(views)
    L = max(len(v[2]) for v in views)
    kmax = max(v[0].k for v in views)
    ids = torch.full((B, L), PAD_ID, dtype=torch.long)
    attn = torch.zeros((B, L), dtype=torch.long)
    mpos = torch.zeros((B, kmax), dtype=torch.long)
    mmask = torch.zeros((B, kmax), dtype=torch.bool)
    gold = torch.full((B,), -1, dtype=torch.long)
    target = torch.zeros((B, kmax), dtype=torch.float32)
    qtype = torch.zeros((B,), dtype=torch.long)
    for b, (item, perm, seq, markers) in enumerate(views):
        ids[b, : len(seq)] = torch.tensor(seq)
        attn[b, : len(seq)] = 1
        qtype[b] = QTYPES[item.qtype]
        for j, p in enumerate(perm):
            target[b, j] = item.target[p]
        if item.gold is not None:
            gold[b] = perm.index(item.gold)
        for j, m in enumerate(markers[: item.k]):
            mpos[b, j] = m
        mmask[b, : len(markers)] = True
    return {"input_ids": ids, "attention_mask": attn, "marker_pos": mpos, "marker_mask": mmask, "gold": gold,
            "target": target, "qtype": qtype}


def forward_options(head_model, batch):
    return head_model(batch["input_ids"], batch["attention_mask"], batch["marker_pos"], batch["marker_mask"], batch["qtype"])


def decision_loss(logits, valid, batch, losses):
    """ce on the gold option (rows that have one) + kl to the teacher target."""
    parts = {}
    has_gold = batch["gold"] >= 0
    if "ce" in losses and bool(has_gold.any()):
        parts["ce"] = ce_loss(logits[has_gold], valid[has_gold], batch["gold"][has_gold])
    if "kl" in losses:
        parts["kl"] = kl_loss(logits, valid, batch["target"])
    total = sum(parts.values()) if parts else logits.new_zeros(())
    return total, {k: float(v) for k, v in parts.items()}


@torch.no_grad()
def evaluate_cross(head_model, tok, items, device, max_len, tokens_per_batch, amp):
    """-> [(item, option logits, option probabilities)] in canonical option order."""
    was_training = head_model.training
    head_model.eval()
    views = [(it, list(range(it.k)), *render(tok, it, list(range(it.k)), max_len)) for it in items]
    order = sorted(range(len(views)), key=lambda i: len(views[i][2]))
    per_item = [None] * len(views)
    i = 0
    while i < len(order):
        L = len(views[order[i]][2])
        bs = max(1, tokens_per_batch // max(L, 1))
        idx = order[i: i + bs]
        i += bs
        batch = {k: v.to(device) for k, v in collate([views[j] for j in idx]).items()}
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits, valid = forward_options(head_model, batch)
        logp = masked_log_softmax(logits.float(), valid)
        for r, j in enumerate(idx):
            it = views[j][0]
            per_item[j] = (it, logits[r, : it.k].float().cpu().tolist(), logp[r, : it.k].exp().cpu().tolist())
    head_model.train(was_training)
    return per_item


# ---------------------------------------------------------------- bi-encoder batches
def batches_by_budget(items_texts, budget, rng, per_option_instr=True):
    """Group rows under a token budget ~ state + options (+ the instruction repeated in every option query)."""
    cost = [min(len(t[0]) // 3, 512) + sum(min(len(c) // 3 + (len(t[1]) // 3 if per_option_instr else 0) + 8, 160) for c in t[2])
            for t in items_texts]
    order = rng.permutation(len(items_texts))
    batches = []
    for start in range(0, len(order), 4096):
        win = sorted(order[start:start + 4096], key=lambda i: cost[i])
        cur, tot = [], 0
        for i in win:
            if cur and tot + cost[i] > budget:
                batches.append(cur); cur, tot = [], 0
            cur.append(int(i)); tot += cost[i]
        if cur:
            batches.append(cur)
    rng.shuffle(batches)
    return batches


@torch.no_grad()
def evaluate_embed(model, tok, items, device, amp, budget=24000):
    """-> [(item, option logits, option probabilities)] in canonical option order."""
    was_training = model.training
    model.eval()
    texts = [texts_for(it.req, it.qtype) for it in items]
    out = [None] * len(items)
    for idx in batches_by_budget(texts, budget, np.random.default_rng(0)):
        b = build_batch(tok, [texts[i] for i in idx], device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            logits, valid = model(b)
        p = torch.log_softmax(logits.float().masked_fill(~valid, -1e4), -1).exp()
        for r, i in enumerate(idx):
            it = items[i]
            out[i] = (it, logits[r, : it.k].cpu().tolist(), p[r, : it.k].cpu().tolist())
    model.train(was_training)
    return out


# ---------------------------------------------------------------- any checkpoint
def predict(path, items, device="cuda", amp=False, tokens_per_batch=16384):
    """Run a trained checkpoint (either kind) on items, fp32 unless amp. -> [(item, logits, probs)]."""
    from transformers import AutoTokenizer
    path = Path(path)
    tok = AutoTokenizer.from_pretrained(path)
    kind = json.loads((path / "decision_config.json").read_text())["head"]
    amp = amp and torch.device(device).type == "cuda"
    if kind == "h2":
        model = load_cross_encoder(path, device)
        out = evaluate_cross(model, tok, items, device, 512, tokens_per_batch, amp)
    else:
        model = EmbedDecisionModel.load(path, device)
        out = evaluate_embed(model, tok, items, device, amp)
    del model
    return out


# ---------------------------------------------------------------- metrics + temperatures
def summarize(per_item):
    n = correct = 0
    nll = brier = 0.0
    tops = []
    by_source, by_type = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    for it, _, p in per_item:
        if it.gold is None:
            continue
        ok = int(np.argmax(p)) == it.gold
        n += 1
        correct += ok
        nll += -math.log(max(p[it.gold], 1e-12))
        brier += sum((p[i] - (1.0 if i == it.gold else 0.0)) ** 2 for i in range(it.k))
        tops.append((max(p), ok))
        by_source[it.source][0] += ok
        by_source[it.source][1] += 1
        by_type[it.qtype][0] += ok
        by_type[it.qtype][1] += 1
    ece = 0.0
    if tops:
        conf = np.array([t[0] for t in tops])
        acc = np.array([t[1] for t in tops], dtype=float)
        edges = np.linspace(0, 1, 11)
        for lo, hi in zip(edges[:-1], edges[1:]):
            sel = (conf > lo) & (conf <= hi)
            if sel.any():
                ece += sel.mean() * abs(conf[sel].mean() - acc[sel].mean())
    return {"n": n, "accuracy": correct / max(1, n), "nll": nll / max(1, n), "brier": brier / max(1, n), "ece": float(ece),
            "by_source": {s: {"n": v[1], "accuracy": v[0] / v[1]} for s, v in sorted(by_source.items())},
            "by_type": {s: {"n": v[1], "accuracy": v[0] / v[1]} for s, v in sorted(by_type.items())}}


def fit_temperatures(per_item, t_min=0.5, t_max=5.0):
    """One temperature per (type, K bucket) minimising NLL on the given logits (golden-section search on log T)."""
    groups = defaultdict(list)
    for it, logits, _ in per_item:
        if it.gold is not None:
            groups[bucket(it.qtype, it.k)].append((np.array(logits), it.gold))

    def nll(g, T):
        s = 0.0
        for z, y in g:
            z = z / T
            z = z - z.max()
            s += -(z[y] - math.log(np.exp(z).sum()))
        return s / len(g)

    out = {}
    for b, g in groups.items():
        lo, hi = math.log(t_min), math.log(t_max)
        gr = (math.sqrt(5) - 1) / 2
        c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
        for _ in range(40):
            if nll(g, math.exp(c)) < nll(g, math.exp(d)):
                hi = d
            else:
                lo = c
            c, d = hi - gr * (hi - lo), lo + gr * (hi - lo)
        T = math.exp((lo + hi) / 2)
        out[b] = {"T": round(T, 4), "n": len(g), "nll_before": round(nll(g, 1.0), 4), "nll_after": round(nll(g, T), 4)}
    return out
