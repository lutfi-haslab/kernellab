"""Referensi softmax PyTorch (eager).

Softmax adalah kernel memory-bound: metrik jujurnya GB/s terhadap puncak
bandwidth T4 (320 GB/s), bukan TFLOPS.
"""
import torch


def softmax_ref(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return torch.softmax(x, dim=dim)
