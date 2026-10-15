"""CLI compatibility entry for baselines.mm_oerfm.run_1d."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from baselines.mm_oerfm.run_1d import main

if __name__ == "__main__":
    main()
