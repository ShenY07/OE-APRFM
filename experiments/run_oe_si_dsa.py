"""CLI compatibility entry for baselines.oe_si_dsa.run."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from baselines.oe_si_dsa.run import main

if __name__ == "__main__":
    main()
