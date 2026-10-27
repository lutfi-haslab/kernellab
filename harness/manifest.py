"""Manifest sesi: identitas satu run benchmark.

Satu manifest JSON per sesi ditulis ke results/sessions/<session_id>.json.
Manifest adalah bagian dari artefak kanonik (commit + fingerprint environment
+ config + versi dataset + versi cache) yang menentukan apakah dua sesi boleh
dibandingkan. Ditulis mesin, tidak pernah diedit tangan.
"""
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return "unknown"


def environment_fingerprint() -> dict:
    env = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        env["torch"] = torch.__version__
        env["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            env["gpu_name"] = torch.cuda.get_device_name(0)
            cap = torch.cuda.get_device_capability(0)
            env["cuda_capability"] = f"{cap[0]}.{cap[1]}"
    except Exception as exc:  # torch tidak ada (mis. sesi TPU) — catat, jangan gagal
        env["torch_error"] = str(exc)
    try:
        import triton

        env["triton"] = triton.__version__
    except Exception:
        pass
    try:
        import jax

        env["jax"] = jax.__version__
        env["jax_devices"] = [str(d) for d in jax.devices()]
    except Exception:
        pass
    return env


def write_manifest(session_id: str, milestone: str, out_dir: str = "results/sessions") -> Path:
    manifest = {
        "session_id": session_id,
        "milestone": milestone,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": _git_commit(),
        "environment": environment_fingerprint(),
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{session_id}.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path
