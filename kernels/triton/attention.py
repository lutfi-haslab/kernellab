"""Attention Triton gaya FlashAttention (GPU T4, FP16) — target >=2x vs eager.

Status: belum diimplementasikan (Minggu 3 rencana KernelLab).
Ditulis manual (tiling KV + online softmax) karena FlashAttention-2 resmi
butuh Ampere+ sedangkan T4 adalah Turing. Spesifikasi beku: forward-only,
B=8, H=12, seq_len 1024, head_dim 64; causal memakai block-skipping
(ablation 5 mengukur efeknya). Toleransi akurasi <=1e-2 vs referensi FP32.
"""


def attention_triton(q, k, v, causal: bool = False):
    raise NotImplementedError("KernelLab Minggu 3: attention Triton")
