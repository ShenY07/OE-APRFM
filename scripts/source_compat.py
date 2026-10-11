"""Compare archived source with current names without changing saved records."""

import ast
import json
import re
from pathlib import Path

NAME_CHANGES = json.loads(
    Path(__file__).with_name("name_changes.json").read_text()
)
NAME_PATTERN = re.compile(
    r"\b(" + "|".join(map(re.escape, NAME_CHANGES)) + r")\b"
)
LAYOUT = json.loads(
    Path(__file__).with_name("layout_changes.json").read_text()
)
PATH_PATTERN = re.compile(
    "|".join(
        re.escape(p) for p in sorted(LAYOUT["paths"], key=len, reverse=True)
    )
)


def rename_source(source):
    """Translate known module and symbol names in source or source paths."""
    if isinstance(source, bytes):
        source = source.decode("utf-8")
    source = NAME_PATTERN.sub(lambda match: NAME_CHANGES[match[0]], source)
    return layout_source(source)


def layout_source(source):
    """Update repository paths and imports after the directory reorganization."""
    source = PATH_PATTERN.sub(lambda match: LAYOUT["paths"][match[0]], source)
    for old, new in LAYOUT["modules"].items():
        for quote in ("'", '"'):
            source = source.replace(quote + old + quote, quote + new + quote)

    def from_import(match):
        module = LAYOUT["modules"].get(match[2], match[2])
        return match[1] + module + match[3]

    source = re.sub(
        r"(^[ \t]*from )([\w.]+)( import\b)",
        from_import,
        source,
        flags=re.M,
    )

    def import_line(match):
        entries = []
        for entry in match[2].split(","):
            parts = entry.strip().split(" as ")
            name = parts[0]
            if name in LAYOUT["modules"]:
                alias = parts[1] if len(parts) == 2 else name.split(".")[0]
                entries.append(LAYOUT["modules"][name] + " as " + alias)
            else:
                entries.append(entry.strip())
        return match[1] + ", ".join(entries)

    return re.sub(
        r"(^[ \t]*import )([^\n#]+)", import_line, source, flags=re.M
    )


def source_tree(source):
    """Normalize names and docstrings while retaining executable statements."""
    tree = ast.parse(rename_source(source))
    for node in ast.walk(tree):
        if (
            isinstance(
                node,
                (
                    ast.Module,
                    ast.ClassDef,
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            )
            and ast.get_docstring(node) is not None
        ):
            node.body.pop(0)
    return tree


def source_ast(source):
    return ast.dump(source_tree(source), include_attributes=False)
