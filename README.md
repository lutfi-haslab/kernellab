# KernelLab

Kernel matmul, softmax, dan attention ditulis tangan di Triton & Pallas, di-benchmark terbuka terhadap cuBLAS dan PyTorch eager di T4.

Satu repo, dua angka. Semua eksekusi berjalan di Kaggle (GPU T4 x2 dan TPU v5e-8); mesin lokal hanya untuk menulis kode. Metodologi, definisi baseline yang dibekukan, dan kriteria sukses/gagal ada di rencana 30 hari (dokumen internal goal) dan diringkas di [REPORT.md](REPORT.md).

## Target vs hasil

| Metrik | Target | Hasil | Status |
|---|---|---|---|
| Matmul Triton @ 4096³ | ≥75% TFLOPS cuBLAS (stretch: 85%) | — | belum diukur |
| Attention Triton @ seq 1024, d64, FP16 | ≥2× vs PyTorch eager | — | belum diukur |
| E2E nanoGPT (prefill, metrik sekunder) | ≥1,15× vs eager | — | belum diukur |

Grafik headline akan tampil di sini setelah run pertama selesai dan terverifikasi di dua sesi Kaggle berbeda.

## Regenerasi laporan

```bash
make report
```

Satu perintah: validasi `results/results.csv` (skema + provenance), lalu render semua grafik ke `plots/out/`. Angka yang tidak bisa diregenerasi dari data mentah tidak masuk laporan.

## Struktur

- `kernels/ref/` — referensi PyTorch (termasuk baseline attention manual eager).
- `kernels/triton/` — kernel GPU (matmul, softmax, attention gaya FlashAttention).
- `kernels/pallas/` — porting attention ke TPU.
- `harness/` — satu pintu timing + correctness, konfigurasi kanonik beku, manifest sesi.
- `notebooks/` — dua notebook kanonik Kaggle (`gpu.ipynb`, `tpu.ipynb`); nama sesi = parameter.
- `results/` — `results.csv` (semua hasil + provenance), `ledger.csv` (satu baris per sesi), `sessions/` (manifest JSON per sesi).
- `plots/` — render grafik dari CSV mentah.

Laporan lengkap: [REPORT.md](REPORT.md) · Data mentah: [results/](results/)
