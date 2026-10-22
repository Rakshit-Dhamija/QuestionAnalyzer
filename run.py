
import subprocess
import sys
from pathlib import Path

# Project root directory
ROOT_DIR = Path(__file__).resolve().parent

# Execute Streamlit using the current Python environment
command = [
    sys.executable,
    "-m",
    "streamlit",
    "run",
    str(ROOT_DIR / "app" / "web-app-new.py"),
]

subprocess.run(command, cwd=ROOT_DIR)
