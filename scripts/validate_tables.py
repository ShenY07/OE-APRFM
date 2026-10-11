"""Cross-check refreshed manuscript entries against saved fields and records."""

import hashlib
import ast
import json
import re
import subprocess
from pathlib import Path
from scripts.source_compat import rename_source, source_ast, source_tree
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/table_refresh_20261010"


def main():
    text = (ROOT / "paper/tables.tex").read_text()
    tables = {
        re.search(r"\\label\{([^}]+)\}", block).group(1): block
        for block in re.findall(
            r"\\begin\{table\}.*?\\end\{table\}", text, re.S
        )
    }
    assert len(tables) == 26
    stack = []
    for match in re.finditer(r"\\(begin|end)\{([^}]+)\}", text):
        if match[1] == "begin":
            stack.append(match[2])
        else:
            assert stack.pop() == match[2]
    assert not stack
    depth = 0
    for i, char in enumerate(text):
        if char in "{}" and (i == 0 or text[i - 1] != "\\"):
            depth += 1 if char == "{" else -1
            assert depth >= 0
    assert depth == 0 and text.count("$") % 2 == 0
    before = (OUT / "table.before_refresh.md").read_text()
    # Preserve concurrently updated APNN rows while refreshing only RF data.
    apnn_pattern = r"OE-APNN\s*&.*?\\\\"
    assert re.findall(apnn_pattern, before, re.S) == re.findall(
        apnn_pattern, text, re.S
    )
    columns = {
        "configurations": 3,
        "mechanism": 5,
        "knudsen-accuracy": 4,
        "efficiency-summary": 6,
        "feature-resolution": 7,
        "parity-budget": 7,
        "p34-components": 7,
        "supp-efficiency-2d": 7,
        "extended": 4,
    }
    cells = {}
    for name, n in columns.items():
        block = tables["tab:" + name]
        body = block.split(r"\midrule", 1)[1].split(r"\bottomrule", 1)[0]
        body = re.sub(r"\\addlinespace(?:\[[^\]]*\])?", "", body)
        rows = []
        for row in re.split(r"\\\\", body):
            if not row.strip() or r"\multicolumn" in row:
                continue
            values = [value.strip() for value in row.split("&")]
            assert len(values) == n, (name, values)
            rows.append(values)
        cells[name] = rows

    def number(cell):
        m = re.search(r"([0-9.]+)\\times10\^\{(-?\d+)\}", cell)
        return float(m[1]) * 10 ** int(m[2]) if m else float(cell)

    common_errors = [
        [number(x) for x in cells["knudsen-accuracy"][-1][2:4]],
        [number(x) for x in cells["efficiency-summary"][-1][3:5]],
        [number(x) for x in cells["feature-resolution"][-1][4:6]],
        [number(x) for x in cells["p34-components"][3][3:5]],
        [number(x) for x in cells["supp-efficiency-2d"][-1][4:6]],
    ]
    assert all(pair == common_errors[0] for pair in common_errors)
    assert (
        number(cells["efficiency-summary"][-1][-1])
        == number(cells["supp-efficiency-2d"][-1][-1])
        == number(cells["p34-components"][3][-1])
    )
    for index, p34row in enumerate((5, 7)):
        assert [number(x) for x in cells["extended"][2 + index][2:4]] == [
            number(x) for x in cells["p34-components"][p34row][3:5]
        ]
    status = json.loads((OUT / "status.json").read_text())
    assert len(status) == 30
    field_checks = []
    for item in status:
        path = ROOT / item["record"]
        r = json.loads(path.read_text())
        with np.load(path.with_suffix(".npz")) as d:
            mask = d["mask"]
            pmask = np.broadcast_to(mask[..., None], d["f"].shape)
            ef = float(
                np.linalg.norm((d["f"] - d["reference_f"])[pmask])
                / np.linalg.norm(d["reference_f"][pmask])
            )
            erho = float(
                np.linalg.norm((d["rho"] - d["reference_rho"])[mask])
                / np.linalg.norm(d["reference_rho"][mask])
            )
            np.testing.assert_allclose(
                d["theta"],
                (np.arange(64) + 0.5) * (2 * np.pi / 64),
                atol=1e-14,
            )
        np.testing.assert_allclose(
            [ef, erho],
            [r["relative_l2_f"], r["relative_l2_rho"]],
            rtol=1e-12,
            atol=1e-15,
        )
        field_checks.append(
            dict(
                source=item["record"],
                Ef_delta=ef - r["relative_l2_f"],
                Erho_delta=erho - r["relative_l2_rho"],
            )
        )
    manifest = json.loads((OUT / "manifest.json").read_text())
    unrelated_source_changes = []
    for path, digest in manifest["sources"].items():
        if (
            path.startswith("src/")
            or rename_source(path) == "experiments/run_p3_oe_aprfm.py"
        ):
            source_path = ROOT / rename_source(path)
            if not source_path.exists():
                # Removed unused variants remain byte-identical in the archive.
                relocations = json.loads(
                    (ROOT / "archive/cleanup_manifest.json").read_text()
                )
                entry = next(
                    item for item in relocations if item["original"] == path
                )
                assert entry["action"] == "move" and entry["sha256"] == digest
                source_path = ROOT / entry["archived"]
                assert (
                    hashlib.sha256(source_path.read_bytes()).hexdigest()
                    == digest
                )
                unrelated_source_changes.append(
                    path + ": unused module archived unchanged"
                )
            current = source_path.read_bytes()
            if (
                rename_source(path) == "experiments/run_p3_oe_aprfm.py"
                and hashlib.sha256(current).hexdigest() != digest
            ):
                original = subprocess.check_output(
                    ["git", "show", manifest["commit"] + ":" + path], cwd=ROOT
                )
                assert hashlib.sha256(original).hexdigest() == digest
                corrected_label = original.replace(
                    b'"independent_pair_traces_v2"',
                    b'"physical_inflow_trace_v3"',
                )
                assert source_ast(corrected_label) == source_ast(current)
                unrelated_source_changes.append(
                    path
                    + ": boundary label and code style updated; numerical logic unchanged"
                )
                continue
            if (
                rename_source(path) == "src/oeaprfm/features/reconstruction.py"
                and hashlib.sha256(current).hexdigest() != digest
            ):
                original = subprocess.check_output(
                    ["git", "show", manifest["commit"] + ":" + path], cwd=ROOT
                )
                assert hashlib.sha256(original).hexdigest() == digest

                def without_half_field(source):
                    tree = source_tree(source)
                    tree.body = [
                        node
                        for node in tree.body
                        if not (
                            isinstance(node, ast.ClassDef)
                            and node.name == "HalfOEField1D"
                        )
                    ]
                    return ast.dump(tree, include_attributes=False)

                assert without_half_field(original) == without_half_field(
                    current
                )
                unrelated_source_changes.append(
                    path + ": only the 1D unprojected constructor changed"
                )
                continue
            if hashlib.sha256(current).hexdigest() != digest:
                original = subprocess.check_output(
                    ["git", "show", manifest["commit"] + ":" + path], cwd=ROOT
                )
                assert hashlib.sha256(original).hexdigest() == digest
                assert source_ast(original) == source_ast(current), path
                unrelated_source_changes.append(
                    path + ": names, comments, and formatting updated"
                )
    result = dict(
        table_count=len(tables),
        checked_table_column_counts=columns,
        cross_table_errors_consistent=True,
        cross_table_times_consistent=True,
        saved_fields_checked=len(field_checks),
        fields=field_checks,
        production_2d_logic_unchanged=True,
        unrelated_source_changes=unrelated_source_changes,
        latex_compilation="Not run: no TeX compiler installed",
    )
    (OUT / "validation.json").write_text(json.dumps(result, indent=2))
    print(
        "Validated 26 tables, 30 saved fields, shared errors/timings, and unchanged 2D production logic."
    )


if __name__ == "__main__":
    main()
