"""Resolve recorded paths without rewriting historical experiment evidence."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MOVES = json.loads(Path(__file__).with_name("folder_changes.json").read_text())


def move_path(source):
    source = str(source)
    for old, new in sorted(MOVES.items(), key=lambda item: -len(item[0])):
        if source == old or source.startswith(old + "/"):
            return new + source[len(old) :]
    return source


def resolve_path(source):
    path = Path(source)
    if path.is_absolute():
        try:
            path = path.relative_to(ROOT)
        except ValueError:
            return path
    return ROOT / move_path(path.as_posix())
