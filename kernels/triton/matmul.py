"""Kernel matmul Triton (GPU T4, FP16) — target >=75% TFLOPS cuBLAS @ 4096^3.

Status: belum diimplementasikan (Minggu 1-2 rencana KernelLab).
Pendekatan: tiling + autotune; konfigurasi dibekukan setelah autotune dan
dicatat di harness/config.py. Ingat T4 = Turing (sm75): tidak ada cp.async,
jadi manfaat num_stages jauh lebih kecil dibanding Ampere — itu diuji eksplisit
di ablation 4.
"""


def matmul_triton(a, b):
    raise NotImplementedError("KernelLab Minggu 2: matmul Triton + autotune")
