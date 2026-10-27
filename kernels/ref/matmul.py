"""Referensi matmul PyTorch — pembandingnya adalah cuBLAS.

torch.matmul di-dispatch PyTorch ke cuBLAS/cuBLASLt, jadi fungsi ini sekaligus
baseline resmi untuk target matmul (TFLOPS absolut + % cuBLAS + % puncak T4).
"""
import torch


def matmul_ref(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.matmul(a, b)
