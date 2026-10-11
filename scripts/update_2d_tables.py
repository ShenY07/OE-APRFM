"""Update every P3/P4 table occurrence from one complete rerun manifest."""

import csv
import hashlib
import json
import math
import re
from pathlib import Path
from scripts.repo_paths import resolve_path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/runs/feature_study"
TARGET = ROOT / "paper/tables.tex"


def sci(value, digits=3, bold=False):
    mantissa, exponent = f"{value:.{digits-1}e}".split("e")
    text = mantissa + r"\times10^{" + str(int(exponent)) + "}"
    return "$" + (r"\boldsymbol{" + text + "}" if bold else text) + "$"


def main():
    statuses = json.loads((OUT / "status.json").read_text())
    assert len(statuses) == 30 and all(
        s["state"] in ("completed", "existing") for s in statuses
    )
    groups = {}
    for status in statuses:
        r = json.loads((resolve_path(status["record"])).read_text())
        assert (
            r["partitions"] == [1, 1, 1]
            and r["collocation"] == [16, 16, 16]
            and r["rcond"] == 1e-12
        )
        r["Tcomp"] = sum(
            r[k]
            for k in ("feature_seconds", "assembly_seconds", "solve_seconds")
        )
        r["rank_fraction"] = r["rank"] / r["num_columns"]
        assert 0 < r["rank"] <= r["num_columns"]
        assert all(
            math.isfinite(r[k]) and r[k] >= 0
            for k in (
                "relative_l2_f",
                "relative_l2_rho",
                "condition_number",
                "Tcomp",
            )
        )
        key = (
            r["problem"],
            r["angular_representation"],
            r["features_per_patch"],
            r["epsilon"],
        )
        groups.setdefault(key, []).append((r, status["record"]))
    summaries = {}
    provenance = []
    for key, members in groups.items():
        assert sorted(r["seed"] for r, p in members) == [11, 23, 37]
        assert len({r["num_rows"] for r, p in members}) == 1
        s = {
            k: median(r[k] for r, p in members)
            for k in (
                "relative_l2_f",
                "relative_l2_rho",
                "condition_number",
                "Tcomp",
                "rank_fraction",
                "num_rows",
                "num_columns",
            )
        }
        summaries[key] = s
        provenance.append(
            dict(
                problem=key[0],
                representation=key[1],
                J=key[2],
                epsilon=key[3],
                **s,
                sources=[p for r, p in members],
            )
        )
    text = TARGET.read_text()
    original = text
    changes = []

    def edit(label, fn):
        nonlocal text
        pattern = re.compile(r"\\begin\{table\}.*?\\end\{table\}", re.S)
        matches = [
            m
            for m in pattern.finditer(text)
            if r"\label{" + label + "}" in m.group()
        ]
        assert len(matches) == 1, label
        m = matches[0]
        new = fn(m.group())
        if new != m.group():
            changes.append(label)
        text = text[: m.start()] + new + text[m.end() :]

    def sub_once(old, new, t):
        assert t.count(old) == 1, old
        return t.replace(old, new, 1)

    def append_caption(t, sentence):
        if sentence in t:
            return t
        start = t.index(r"\caption{") + len(r"\caption{")
        depth = 1
        for i in range(start, len(t)):
            if t[i] in "{}" and t[i - 1] != "\\":
                depth += 1 if t[i] == "{" else -1
                if depth == 0:
                    return t[:i] + " " + sentence + t[i:]
        raise ValueError("Unbalanced caption")

    def p3(J=128, e=0.001):
        return summaries["p3", "four_component", J, e]

    def errors(s, d=4):
        return sci(s["relative_l2_f"], d) + " & " + sci(s["relative_l2_rho"])

    def accuracy(t):
        start = t.index("2D manufactured &")
        end = t.index(r"\bottomrule", start)
        return (
            t[:start]
            + (
                "2D manufactured & $1$ & "
                + errors(p3(e=1.0))
                + r"\\"
                + "\n & $10^{-3}$ & "
                + errors(p3())
                + r"\\"
                + "\n"
            )
            + t[end:]
        )

    edit("tab:knudsen-accuracy", accuracy)

    def efficiency(t):
        start = t.index(r"   & \textbf{OE-APRFM} & 512")
        end = t.index(r"\bottomrule", start)
        t = (
            t[:start]
            + r"   & \textbf{OE-APRFM} & 512"
            + "\n   & "
            + errors(p3())
            + f" & {p3()['Tcomp']:.2f} "
            + r"\\"
            + "\n"
            + t[end:]
        )
        t = t.replace(
            "historical timings of other methods", "timings of other methods"
        )
        return append_caption(
            t,
            "The 2D OE-APRFM entries are from the unified-protocol reruns; their computation times are recorded serial fresh-process measurements on a shared host.",
        )

    edit("tab:efficiency-summary", efficiency)

    def feature(t):
        start = t.index("2D & 32 &")
        end = t.index(r"\bottomrule", start)
        rows = []
        for J in (32, 64, 128):
            s = p3(J)
            rank = (
                "1.000"
                if s["rank_fraction"] == 1
                else f"{s['rank_fraction']:.6f}"
            )
            rows.append(
                f"{'2D' if J==32 else ''} & {J} & {int(s['num_columns'])} & {int(s['num_rows'])} & "
                + sci(s["relative_l2_f"], 4, J == 128)
                + " & "
                + sci(s["relative_l2_rho"], 3, J == 128)
                + f" & {rank} "
                + r"\\"
            )
        return t[:start] + "\n".join(rows) + "\n" + t[end:]

    edit("tab:feature-resolution", feature)

    def components(t):
        a = t.index(r"\caption{")
        b = t.index(r"\label{", a)
        t = (
            t[:a]
            + r"\caption{Two- and four-component angular representations at 512 coefficients. All entries are medians over seeds 11, 23 and 37 from fresh-process reruns of the current implementation, using nominal $16\times16\times16$ collocation parameters and a common angular-midpoint evaluation grid. The four-component runs use physical-inflow trace assembly. Times include feature construction, assembly and solution, but exclude evaluation; runs were serialized on a shared host.}"
            + "\n"
            + t[b:]
        )
        start = t.index(r"\multicolumn{7}{l}{2D manufactured solution}")
        end = t.index(r"\bottomrule", start)
        rows = []
        for problem, title in [
            ("p3", "2D manufactured solution"),
            ("p4", "Perforated square"),
        ]:
            if rows:
                rows.append(r"\addlinespace")
            rows.append(r"\multicolumn{7}{l}{" + title + r"}\\")
            for eps in (1.0, 0.001):
                for components, representation, J in [
                    (2, "legacy_two_component", 256),
                    (4, "four_component", 128),
                ]:
                    s = summaries[problem, representation, J, eps]
                    e = (
                        ("$1$" if eps == 1.0 else "$10^{-3}$")
                        if components == 2
                        else ""
                    )
                    rows.append(
                        f"{e} & {components} & {int(s['num_rows'])} & "
                        + errors(s)
                        + " & "
                        + sci(s["condition_number"])
                        + f" & {s['Tcomp']:.2f}"
                        + r"\\"
                    )
        return t[:start] + "\n".join(rows) + "\n" + t[end:]

    edit("tab:p34-components", components)

    def costs(t):
        start = t.index("OE-APRFM & $J=32$")
        end = t.index(r"\bottomrule", start)
        rows = []
        for J in (32, 64, 128):
            s = p3(J)
            rows.append(
                ("OE-APRFM" if J == 32 else "")
                + f" & $J={J}$ & {int(s['num_columns'])} & {int(s['num_rows'])} & "
                + errors(s)
                + f" & {s['Tcomp']:.2f}"
                + r"\\"
            )
        t = t[:start] + "\n".join(rows) + "\n" + t[end:]
        t = t.replace(
            "other-method times are historical",
            "deterministic and MM times are historical",
        )
        return append_caption(
            t,
            "OE-APRFM uses physical-inflow trace assembly and angular-midpoint evaluation; its errors and recorded CPU computation times are updated together from serial fresh-process reruns on a shared host.",
        )

    edit("tab:supp-efficiency-2d", costs)

    def extended(t):
        start = t.index("Perforated square &")
        end = t.index(r"\bottomrule", start)
        rows = []
        for e in (1.0, 0.001):
            s = summaries["p4", "four_component", 128, e]
            rows.append(
                ("Perforated square & $1$" if e == 1.0 else " & $10^{-3}$")
                + " & "
                + errors(s)
                + r"\\"
            )
        return t[:start] + "\n".join(rows) + "\n" + t[end:]

    edit("tab:extended", extended)

    assert text.count(r"\begin{table}") == text.count(r"\end{table}") == 25
    assert "15856" not in text and "15,856" not in text
    assert "separate numerical protocols" not in text
    assert "timing entries are not reported" not in text
    assert (
        TARGET.read_text() == original
    ), "Table changed during update; retry against the latest file."
    TARGET.write_text(text)
    (OUT / "table.before_refresh.md").write_text(original)
    (OUT / "table.after_refresh.md").write_text(text)
    (OUT / "table_provenance.json").write_text(
        json.dumps(
            dict(
                target=str(TARGET.relative_to(ROOT)),
                updater_sha256=hashlib.sha256(
                    Path(__file__).read_bytes()
                ).hexdigest(),
                sha256=hashlib.sha256(text.encode()).hexdigest(),
                changed_labels=changes,
                groups=provenance,
            ),
            indent=2,
        )
    )
    with (OUT / "summary.csv").open("w") as f:
        w = csv.DictWriter(
            f, fieldnames=[k for k in provenance[0] if k != "sources"]
        )
        w.writeheader()
        w.writerows(
            {k: v for k, v in r.items() if k != "sources"} for r in provenance
        )
    print("Updated", ", ".join(changes))


if __name__ == "__main__":
    main()
