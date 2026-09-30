"""FLAN Answer Token Prediction data (written by prepare_flan.py): token pool, length-grouped batches, masking.

Layout of every example: [CLS] inputs " " [unused0] target [SEP], the single answer token at position n-2.
Used by train_atp.py (instruction tuning of B's backbone) and by train_decision.py (FLAN replay).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

MASK_ID, PAD_ID, CLS_ID, SEP_ID, ANCHOR_ID = 50284, 50283, 50281, 50282, 50285


class Pool:
    def __init__(self, prefix: str):
        self.tokens = np.load(prefix + ".tokens.npy", mmap_mode="r")
        self.offsets = np.load(prefix + ".offsets.npy")
        self.meta = json.loads(Path(prefix + ".meta.json").read_text())

    def __len__(self):
        return len(self.offsets) - 1

    def get(self, i: int) -> np.ndarray:
        return np.asarray(self.tokens[self.offsets[i]:self.offsets[i + 1]])

    def lengths(self, n):
        return np.diff(self.offsets[: n + 1])


def make_batches(lengths: np.ndarray, tokens_per_batch: int, rng: np.random.Generator, group: int = 64):
    """Length-grouped batches: sort inside windows of `group` batches worth of examples, then shuffle batch order."""
    n = len(lengths)
    order = rng.permutation(n)
    mean_len = max(1.0, float(lengths.mean()))
    window = int(tokens_per_batch / mean_len) * group
    batches = []
    for start in range(0, n, window):
        idx = order[start:start + window]
        idx = idx[np.argsort(lengths[idx], kind="stable")]
        cur, cur_max = [], 0
        for i in idx:
            L = int(lengths[i])
            new_max = max(cur_max, L)
            if cur and new_max * (len(cur) + 1) > tokens_per_batch:
                batches.append(cur); cur, cur_max = [], 0
                new_max = L
            cur.append(int(i)); cur_max = new_max
        if cur:
            batches.append(cur)
    perm = rng.permutation(len(batches))
    return [batches[i] for i in perm]


def collate(pool: Pool, idx: list[int], rng: np.random.Generator, atp_frac: float, mlm_prob: float, eval_mode=False):
    """Per example, with probability atp_frac: the answer token is replaced by [MASK] and is the only label (ATP);
    otherwise "dummy MLM": mlm_prob of the non-special tokens are masked and labelled with the [MASK] id itself."""
    seqs = [pool.get(i) for i in idx]
    L = max(len(s) for s in seqs)
    B = len(seqs)
    input_ids = np.full((B, L), PAD_ID, dtype=np.int64)
    attn = np.zeros((B, L), dtype=np.int64)
    labels = np.full((B, L), -100, dtype=np.int64)
    is_atp = np.zeros(B, dtype=bool)
    for b, s in enumerate(seqs):
        n = len(s)
        input_ids[b, :n] = s
        attn[b, :n] = 1
        tpos = n - 2
        assert s[tpos - 1] == ANCHOR_ID and s[n - 1] == SEP_ID, "unexpected sequence layout"
        if eval_mode or rng.random() < atp_frac:
            labels[b, tpos] = s[tpos]
            input_ids[b, tpos] = MASK_ID
            is_atp[b] = True
        else:
            special = (s == CLS_ID) | (s == SEP_ID) | (s == PAD_ID)
            m = (rng.random(n) < mlm_prob) & ~special
            input_ids[b, :n][m] = MASK_ID
            labels[b, :n][m] = MASK_ID  # dummy label: the mask token itself
    return (torch.from_numpy(input_ids), torch.from_numpy(attn), torch.from_numpy(labels), torch.from_numpy(is_atp))


def lr_at(step, total, warmup, peak):
    """Linear warmup, then linear decay to 0."""
    if step < warmup:
        return peak * step / max(1, warmup)
    return peak * max(0.0, (total - step) / max(1, total - warmup))
