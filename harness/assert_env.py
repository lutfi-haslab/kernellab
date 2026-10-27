"""Gerbang awal sesi GPU (protokol KernelLab: assert dalam 5 menit pertama).

Menjalankan gerbang 15 menit sesi gpu-w1-s1: pastikan accelerator adalah
T4/sm75 — sesi yang mendarat di P100/sm60 atau hardware lain DIBATALKAN,
karena Triton tidak berjalan di sana. Bila lolos, tulis manifest sesi.

Pakai:  python harness/assert_env.py --session gpu-w1-s1-env-assert --milestone M1
Keluar dengan kode non-nol bila assert gagal (sesi tidak sah, jangan benchmark).
"""
import argparse
import sys

sys.path.insert(0, ".")  # repo root, agar harness.* bisa diimpor dari notebook Kaggle

from harness.config import EXPECTED_CUDA_CAPABILITY
from harness.manifest import write_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", required=True)
    parser.add_argument("--milestone", required=True)
    args = parser.parse_args()

    import torch

    if not torch.cuda.is_available():
        print("ABORT: CUDA tidak tersedia — sesi ini bukan sesi GPU.")
        return 1
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    if cap != EXPECTED_CUDA_CAPABILITY:
        print(f"ABORT: GPU {name} capability {cap}, target T4 {EXPECTED_CUDA_CAPABILITY}. "
              "Tutup sesi ini (maks 5 menit) dan buka sesi baru.")
        return 1
    path = write_manifest(args.session, args.milestone)
    print(f"OK: {name} sm{cap[0]}{cap[1]} — manifest ditulis ke {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
