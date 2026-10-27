# Laporan Benchmark KernelLab

> Status: kerangka. Semua angka masih **belum diukur** — section terisi setelah run headline Minggu ke-4, terverifikasi di dua sesi Kaggle berbeda.

## 1. Headline & vonis

| Angka headline | Target | Hasil | Status |
|---|---|---|---|
| Matmul Triton @ 4096³ (% TFLOPS cuBLAS) | ≥75% (stretch 85%) | — | belum diukur |
| Attention Triton @ seq 1024 (× vs eager) | ≥2× | — | belum diukur |

## 2. Metodologi & lingkungan uji

- Baseline dibekukan sebelum pengukuran pertama: attention eager = manual (QK^T → scale → softmax → PV, skor termaterialisasi), forward-only, B=8, H=12, head_dim 64, FP16; causal dan non-causal dilaporkan terpisah.
- Timing: `triton.testing.do_bench` (warmup dibuang, median dari N run; sebaran p10/p90 dicatat).
- Toleransi akurasi: max abs error ≤1e-2 vs referensi FP32. Kernel yang gagal akurasi tidak diukur kecepatannya.
- Lingkungan: Kaggle, GPU T4 (puncak FP16 65 TFLOPS, bandwidth 320 GB/s) dan TPU v5e-8. Versi torch/triton/jax, driver, dan commit hash tercatat per baris di `results/results.csv`; manifest per sesi di `results/sessions/`.
- Batasan jujur: sesi Kaggle bersama (shared), bukan mesin dedicated — karena itu angka headline wajib stabil di dua sesi berbeda (lihat grafik sebaran antar-sesi).

## 3. Hasil matmul & softmax

_Belum diukur._ Grafik: TFLOPS vs shape (garis puncak 65 TFLOPS, label % cuBLAS di 4096³) dan rasio vs cuBLAS (garis target 0,75, stretch 0,85). Softmax dilaporkan sebagai tabel GB/s terhadap puncak 320 GB/s.

## 4. Hasil attention

_Belum diukur._ Grafik speedup vs eager per seq_len (causal & non-causal seri terpisah); tabel e2e nanoGPT (prefill, metrik sekunder, ambang ≥1,15×) di section ini.

## 5. Ablation

_Belum diukur._ Enam ablation wajib: (1) tiling on/off; (2) online softmax vs dua-pass; (3) fusion on/off; (4) tile/stage default vs autotune terbaik di T4; (5) causal block-skipping vs dihitung penuh; (6) akumulator FP32 vs FP16. Grafik sebaran antar-sesi untuk dua angka headline juga tampil di sini.

## 6. Kegagalan & batasan

_Akan diisi apa adanya:_ shape di mana kernel kalah dari cuBLAS/eager, target stretch yang tidak tercapai, dan batasan metodologi. Section ini tidak akan dikosongkan — laporan yang hanya berisi kemenangan bukan laporan engineering.
