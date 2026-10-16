"""Start FastAPI (internal) + Streamlit (public $PORT) for cloud/local."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
os.environ["PYTHONPATH"] = str(ROOT) + os.pathsep + os.environ.get("PYTHONPATH", "")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PORT = os.environ.get("PORT", "8501")
API_HOST = os.environ.get("API_HOST", "127.0.0.1")
API_PORT = os.environ.get("API_PORT", "8000")
os.environ.setdefault("API_URL", f"http://{API_HOST}:{API_PORT}")


def main() -> None:
    api = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "backend.main:app",
            "--host",
            API_HOST,
            "--port",
            API_PORT,
        ],
        cwd=str(ROOT),
    )
    # brief wait so model loads before UI hits API
    time.sleep(3)
    ui = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "streamlit",
            "run",
            "frontend/app.py",
            "--server.port",
            str(PORT),
            "--server.address",
            "0.0.0.0",
            "--server.headless",
            "true",
            "--browser.gatherUsageStats",
            "false",
        ],
        cwd=str(ROOT),
        env=os.environ.copy(),
    )
    try:
        ui.wait()
    finally:
        api.terminate()
        ui.terminate()


if __name__ == "__main__":
    main()