"""Porting attention ke Pallas (TPU v5e-8) — tempat porting, bukan eksperimen.

Status: belum diimplementasikan (gerbang hello-world Minggu 3, porting Minggu 4).
Baseline: referensi JAX murni (matmul + softmax) di TPU yang sama; laporkan ms
dan rasio. Runtime TPU Kaggle tanpa internet: versi JAX/Pallas mengikuti image
Kaggle dan kode dibawa sebagai dataset input, bukan git clone.
"""


def attention_pallas(q, k, v, causal: bool = False):
    raise NotImplementedError("KernelLab Minggu 4: porting attention Pallas")
