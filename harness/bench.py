"""Satu pintu timing + correctness untuk semua kernel KernelLab.

Kontrak (diimplementasikan Minggu 1, sesi gpu-w1-s2-harness-freeze):
- correctness dulu: output kernel dibandingkan ke referensi FP32; max abs error
  di atas harness.config.TOLERANCE_MAX_ABS_ERR berarti baris hasil ditolak.
- timing memakai triton.testing.do_bench (warmup dibuang, median dilaporkan,
  p10/p90 dicatat) — bukan time.time() manual.
- setiap pengukuran menghasilkan tepat satu baris results/results.csv lengkap
  dengan provenance (session_id, git_commit, versi library, gpu_name).
"""


def bench_kernel(*args, **kwargs):
    raise NotImplementedError("KernelLab Minggu 1: harness benchmark (sesi gpu-w1-s2)")
