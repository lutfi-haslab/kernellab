"""Kernel softmax Triton (GPU T4) — jembatan menuju attention.

Status: belum diimplementasikan (Minggu 2 rencana KernelLab).
Dua varian yang akan dibandingkan (ablation 2 & 3):
- dua-pass di dalam kernel (max, lalu sum, lalu normalize) sebagai kernel utuh;
- online softmax (running max + running sum) — pola yang sama yang dipakai
  attention gaya FlashAttention agar skor tidak pernah termaterialisasi.
"""


def softmax_triton(x):
    raise NotImplementedError("KernelLab Minggu 2: softmax dua-pass + online")
