"""E2E: nanoGPT mini satu file, vendored — tanpa dependensi repo luar.

Status: belum diimplementasikan (Minggu 4 rencana KernelLab).
Dipakai hanya untuk metrik sekunder: prefill/forward throughput (tokens/detik)
pada konteks 1024, ambang >=1,15x vs versi attention eager. Mode decode tidak
dipakai — di sana bottleneck-nya bobot model, bukan attention.
"""


def build_model():
    raise NotImplementedError("KernelLab Minggu 4: nanoGPT mini untuk e2e prefill")
