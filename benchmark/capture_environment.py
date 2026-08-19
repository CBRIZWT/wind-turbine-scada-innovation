from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


OUTPUT = Path(r"E:\创新\论文复现\复现基准_2026-07-19\environment")


def run(command: list[str]) -> dict[str, object]:
    try:
        completed = subprocess.run(
            command, text=True, encoding="utf-8", errors="replace",
            capture_output=True, check=False, timeout=120,
        )
        return {"command": command, "returncode": completed.returncode,
                "stdout": completed.stdout, "stderr": completed.stderr}
    except Exception as exc:  # 环境审计失败不能伪装成成功
        return {"command": command, "returncode": None, "stdout": "",
                "stderr": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    pip = run([sys.executable, "-m", "pip", "freeze", "--all"])
    (OUTPUT / "pip_freeze.txt").write_text(str(pip["stdout"]), encoding="utf-8")

    conda_exe = Path(sys.executable).resolve().parents[1] / "Scripts" / "conda.exe"
    conda = run([str(conda_exe), "list", "--explicit"]) if conda_exe.is_file() else {
        "command": [str(conda_exe), "list", "--explicit"], "returncode": None,
        "stdout": "", "stderr": "conda_executable_not_found",
    }
    (OUTPUT / "conda_explicit.txt").write_text(str(conda["stdout"]), encoding="utf-8")
    smi_exe = shutil.which("nvidia-smi")
    smi = run([smi_exe, "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]) if smi_exe else {
        "command": ["nvidia-smi"], "returncode": None, "stdout": "", "stderr": "nvidia_smi_not_found",
    }

    torch_info: dict[str, object]
    try:
        import torch
        torch_info = {
            "version": torch.__version__, "cuda_version": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "cudnn_version": torch.backends.cudnn.version(),
            "devices": [
                {"index": i, "name": torch.cuda.get_device_name(i),
                 "capability": list(torch.cuda.get_device_capability(i))}
                for i in range(torch.cuda.device_count())
            ],
        }
    except Exception as exc:
        torch_info = {"status": "import_failed", "error": f"{type(exc).__name__}: {exc}"}

    disk = shutil.disk_usage("E:\\")
    manifest = {
        "interpreter": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "torch": torch_info,
        "nvidia_smi": smi,
        "pip_freeze": {"returncode": pip["returncode"], "stderr": pip["stderr"]},
        "conda_explicit": {"returncode": conda["returncode"], "stderr": conda["stderr"]},
        "disk_e_bytes": {"total": disk.total, "used": disk.used, "free": disk.free},
    }
    encoded = json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False, sort_keys=True) + "\n"
    (OUTPUT / "environment.json").write_text(encoded, encoding="utf-8")
    digest = hashlib.sha256(
        (encoded + str(pip["stdout"]) + str(conda["stdout"])).encode("utf-8")
    ).hexdigest()
    (OUTPUT / "environment.sha256").write_text(digest + "\n", encoding="ascii")
    print(json.dumps({"output": str(OUTPUT), "environment_hash": digest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
