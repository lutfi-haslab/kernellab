"""Render semua grafik laporan dari results/results.csv — satu-satunya pintu.

`make report` menjalankan file ini dengan --check:
1. validasi skema results.csv + provenance wajib tidak boleh kosong;
2. render 4 grafik ke plots/out/ (matmul, attention, ablation, sebaran sesi).

Status saat repo dibuat: langkah 1 (validasi) aktif; langkah 2 (render)
diimplementasikan Minggu 4 bersama data pertama. Angka yang tidak bisa
diregenerasi dari data mentah tidak masuk laporan.
"""
import argparse
import csv
import sys
from pathlib import Path

REQUIRED_COLUMNS = [
    "timestamp_iso", "session_id", "run_id", "git_commit", "family", "kernel",
    "backend", "variant", "shape_m", "shape_n", "shape_k", "batch", "heads",
    "seq_len", "head_dim", "causal", "dtype", "accum_dtype", "time_ms_median",
    "time_ms_p10", "time_ms_p90", "n_runs", "tflops", "gbps",
    "baseline_time_ms", "ratio_vs_baseline", "gpu_name", "gpu_peak_tflops_fp16",
    "gpu_peak_gbps", "torch_version", "triton_version", "jax_version",
    "cuda_version", "kaggle_kernel_slug",
]
PROVENANCE_COLUMNS = ["session_id", "git_commit", "gpu_name"]


def validate(csv_path: Path) -> int:
    if not csv_path.exists():
        print(f"GAGAL: {csv_path} tidak ditemukan")
        return 1
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != REQUIRED_COLUMNS:
            print(f"GAGAL: skema kolom tidak cocok.\n  ada : {reader.fieldnames}\n  wajib: {REQUIRED_COLUMNS}")
            return 1
        rows = list(reader)
    for i, row in enumerate(rows, start=2):
        for col in PROVENANCE_COLUMNS:
            if not row[col].strip():
                print(f"GAGAL: baris {i} kolom provenance '{col}' kosong — baris ditolak")
                return 1
    print(f"OK: {csv_path} valid, {len(rows)} baris hasil.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="validasi CSV (render menyusul Minggu 4)")
    parser.add_argument("--csv", default="results/results.csv")
    args = parser.parse_args()
    code = validate(Path(args.csv))
    if code == 0:
        print("Render grafik: belum ada data — diimplementasikan Minggu 4 (plots/out/).")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
