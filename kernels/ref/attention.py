"""Baseline attention manual eager — definisinya DIBEKUKAN (lihat REPORT.md).

QK^T -> scale -> (mask causal) -> softmax -> PV, dengan matriks skor
termaterialisasi penuh di HBM. Forward-only. Inilah yang harus dikalahkan
kernel Triton >=2x pada seq_len 1024, head_dim 64, FP16, B=8, H=12;
causal dan non-causal dilaporkan terpisah.

Jangan mengganti implementasi ini dengan SDPA/xFormers: itu mengubah definisi
baseline dan membuat angka speedup tidak falsifiable. Angka SDPA boleh dicatat
sebagai konteks di laporan, bukan sebagai baseline.
"""
import math

import torch


def attention_eager_ref(
    q: torch.Tensor,  # (B, H, S, D)
    k: torch.Tensor,  # (B, H, S, D)
    v: torch.Tensor,  # (B, H, S, D)
    causal: bool = False,
) -> torch.Tensor:
    d = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d)
    if causal:
        s = q.shape[-2]
        mask = torch.triu(
            torch.ones(s, s, device=q.device, dtype=torch.bool), diagonal=1
        )
        scores = scores.masked_fill(mask, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)
