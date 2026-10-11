"""Check the paper snapshot and optionally package its numbered PDF panels."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def check_paper(paper):
    manifest = json.loads((paper / "manifest.json").read_text())
    records = manifest["files"]
    paths = [row["path"] for row in records]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate paths in the paper manifest")
    for row in records:
        path = paper / row["path"]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != row["sha256"]:
            raise ValueError(f"Paper snapshot changed: {row['path']}")
    text = (paper / "figures.tex").read_text()
    panels = re.findall(r"\\includegraphics\[[^\]]*\]\s*\{([^}]+)\}", text)
    recorded = {path for path in paths if path.endswith(".pdf")}
    if set(panels) != recorded or len(panels) != len(recorded):
        raise ValueError("Figure references do not match the manifest")
    tables = (paper / "tables.tex").read_text()
    labels = re.findall(r"\\label\{([^}]+)\}", text + tables)
    if len(labels) != len(set(labels)):
        raise ValueError("Duplicate LaTeX labels in the manuscript")
    return panels


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", action="store_true")
    args = parser.parse_args()
    paper = ROOT / "paper"
    panels = check_paper(paper)
    if args.bundle:
        with zipfile.ZipFile(paper / "figures.zip", "w") as bundle:
            for name in panels:
                entry = zipfile.ZipInfo(name)
                entry.compress_type = zipfile.ZIP_DEFLATED
                bundle.writestr(entry, (paper / name).read_bytes())
    print(f"Verified the paper manifest and {len(panels)} PDF panels.")


if __name__ == "__main__":
    main()
