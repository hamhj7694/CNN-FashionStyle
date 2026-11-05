"""CPU deployment entrypoint; local training requirements remain separate."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import main

main()
