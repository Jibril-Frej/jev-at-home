"""Supervised objectives over K option logits."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_log_softmax(logits, valid):
    return F.log_softmax(logits.masked_fill(~valid, -1e4), -1)


def ce_loss(logits, valid, gold):
    """Cross-entropy on the gold option."""
    return F.nll_loss(masked_log_softmax(logits, valid), gold)


def kl_loss(logits, valid, target):
    """KL(target || student) over the valid options. target: [B, K] probabilities (zeros beyond K)."""
    logp = masked_log_softmax(logits, valid)
    t = torch.where(valid, target.clamp_min(1e-8), torch.ones_like(target))  # padded slots: log(1) = 0, masked below
    return torch.where(valid, t * (torch.log(t) - logp), torch.zeros_like(t)).sum(-1).mean()
