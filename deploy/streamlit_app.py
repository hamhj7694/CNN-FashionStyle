"""CPU deployment entrypoint; local training requirements remain separate."""
import os
from pathlib import Path
import runpy
import sys

# Shared cloud CPUs can expose more host cores than this app can actually use.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import torch

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Execute the page on every Streamlit rerun, including its top-level CSS.
runpy.run_path(str(ROOT / "app.py"), run_name="__main__")
