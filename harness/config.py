"""Konfigurasi kanonik KernelLab — DIBEKUKAN di akhir Minggu 1.

Perubahan file ini setelah beku = commit tersendiri dengan alasan, dan
menciptakan generasi benchmark baru (baseline wajib diulang). Jangan dicampur
dengan commit kernel.
"""

# Toleransi akurasi: kernel yang error-nya di atas ini tidak diukur kecepatannya.
TOLERANCE_MAX_ABS_ERR = 1e-2  # FP16 vs referensi FP32

# Hardware target (Kaggle).
GPU_PEAK_TFLOPS_FP16 = 65.0   # T4
GPU_PEAK_GBPS = 320.0         # T4
EXPECTED_CUDA_CAPABILITY = (7, 5)  # T4 / sm75

# Shape sweep matmul (cukup tiga — memotong shape, bukan ablation, bila waktu sempit).
MATMUL_SHAPES = [(1024, 1024, 1024), (2048, 2048, 2048), (4096, 4096, 4096)]

# Spesifikasi attention yang dibekukan (baseline & kernel memakai ini persis).
ATTENTION = {
    "batch": 8,
    "heads": 12,
    "seq_len": 1024,
    "head_dim": 64,
    "dtype": "float16",
    "forward_only": True,
}

# Target (tervalidasi — lihat REPORT.md untuk metodologi).
TARGETS = {
    "matmul_ratio_vs_cublas": 0.75,     # target utama; 0.85 = stretch
    "matmul_stretch_ratio": 0.85,
    "matmul_kill_ratio_day14": 0.60,    # di bawah ini scope diformat ulang
    "attention_speedup_vs_eager": 2.0,
    "e2e_prefill_speedup": 1.15,        # metrik sekunder
}
