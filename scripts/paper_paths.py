"""Resolve manuscript assets using their recorded source paths."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"


def paper_file(source):
    source = Path(source)
    if source.is_absolute():
        source = source.relative_to(ROOT)
    manifest = PAPER / "manifest.json"
    if not manifest.exists():
        return ROOT / source
    records = json.loads(manifest.read_text())["files"]
    paths = {row["source"]: row["path"] for row in records}
    key = source.as_posix()
    if key in paths:
        return PAPER / paths[key]
    if (
        source.suffix == ".png"
        and source.with_suffix(".pdf").as_posix() in paths
    ):
        return (
            PAPER / paths[source.with_suffix(".pdf").as_posix()]
        ).with_suffix(".png")
    return ROOT / source
