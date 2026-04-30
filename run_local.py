from __future__ import annotations

import argparse
import os
import subprocess
import urllib.request
import venv
from pathlib import Path


ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "best.pt"
MODEL_URL = "https://huggingface.co/DefendIntelligence/vessel-detection/resolve/main/models/best.pt"


def _venv_python() -> Path:
    if os.name == "nt":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"


def _run(command: list[str | os.PathLike[str]], env: dict[str, str] | None = None) -> None:
    printable = " ".join(str(part) for part in command)
    print(f"\n$ {printable}", flush=True)
    subprocess.check_call([str(part) for part in command], cwd=ROOT, env=env)


def _ensure_venv() -> Path:
    python_path = _venv_python()
    if not python_path.exists():
        print(f"Creating virtual environment: {VENV_DIR}", flush=True)
        venv.EnvBuilder(with_pip=True).create(VENV_DIR)
    return python_path


def _install_dependencies(python_path: Path) -> None:
    _run([python_path, "-m", "pip", "install", "--upgrade", "pip"])
    _run([python_path, "-m", "pip", "install", "-r", "requirements.txt"])


def _download_model() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 0:
        print(f"Model already present: {MODEL_PATH}", flush=True)
        return

    tmp_path = MODEL_PATH.with_suffix(".pt.tmp")
    print(f"Downloading model from Hugging Face:\n{MODEL_URL}", flush=True)
    with urllib.request.urlopen(MODEL_URL) as response, tmp_path.open("wb") as handle:
        total = int(response.headers.get("Content-Length") or 0)
        downloaded = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
            downloaded += len(chunk)
            if total:
                percent = downloaded * 100 / total
                print(f"\r{downloaded / 1_000_000:.1f} MB / {total / 1_000_000:.1f} MB ({percent:.0f}%)", end="")
            else:
                print(f"\r{downloaded / 1_000_000:.1f} MB", end="")
        print()
    tmp_path.replace(MODEL_PATH)
    print(f"Saved model to: {MODEL_PATH}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Install and run the Vessel Detection Gradio demo locally.")
    parser.add_argument("--skip-install", action="store_true", help="Do not install Python dependencies.")
    parser.add_argument("--download-only", action="store_true", help="Download the model and exit.")
    parser.add_argument("--host", default="127.0.0.1", help="Gradio server host.")
    parser.add_argument("--port", default="7860", help="Gradio server port.")
    args = parser.parse_args()

    python_path = None
    if not (args.download_only and args.skip_install):
        python_path = _ensure_venv()
    if not args.skip_install:
        if python_path is None:
            python_path = _ensure_venv()
        _install_dependencies(python_path)
    _download_model()

    if args.download_only:
        return

    if python_path is None:
        python_path = _ensure_venv()
    env = os.environ.copy()
    env["GRADIO_SERVER_NAME"] = args.host
    env["GRADIO_SERVER_PORT"] = args.port
    print(f"\nStarting Gradio at http://{args.host}:{args.port}", flush=True)
    _run([python_path, "app.py"], env=env)


if __name__ == "__main__":
    main()
