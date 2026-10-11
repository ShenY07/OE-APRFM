"""Verify that curated research files match their recorded source bytes."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    records = json.loads((ROOT / "data/manifest.json").read_text())
    for record in records:
        path = ROOT / record["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != record["sha256"]:
            raise ValueError(f"Asset differs from recorded source: {path}")
    print(f"Verified {len(records)} research assets")


if __name__ == "__main__":
    main()
