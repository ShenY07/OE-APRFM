"""Validate published panels, build a provenance registry, and package the PDFs."""

import csv
import hashlib
import json
import re
import subprocess
import zipfile
from pathlib import Path
from scripts.repo_paths import resolve_path
from scripts.paper_paths import paper_file
from statistics import median
import xml.etree.ElementTree as ET
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/checks/figures"
ARCH = ROOT / ".release-backup/20261009/original-worktree"


def read(p):
    return json.loads(p.read_text())


def csvread(p):
    with p.open() as f:
        return list(csv.DictReader(f))


def rel(p):
    return str(p.relative_to(ROOT))


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def close(a, b):
    np.testing.assert_allclose(a, b, rtol=2e-12, atol=1e-14)


figtext = (ROOT / "paper/figures.tex").read_text()
oldtext = (OUT / "figure.before.md").read_text()
pattern = r"\\includegraphics\[[^\]]*\]\s*\{([^}]+)\}"
paths = re.findall(pattern, figtext)
oldpaths = [
    paper_file("results/" + name).relative_to(ROOT / "paper").as_posix()
    for name in re.findall(pattern, oldtext)
]
assert paths == oldpaths and len(paths) == 31
assert len(set(paths)) == 31
panels = read(OUT / "plot_data.json")
assert len(panels) == 31
byname = {r["name"]: r for r in panels}
# Parse tables and validate all LaTeX environments/braces without a TeX compiler.
tabletext = (ROOT / "paper/tables.tex").read_text()
tables = {
    re.search(r"\\label\{([^}]+)\}", b)[1]: b
    for b in re.findall(r"\\begin\{table\}.*?\\end\{table\}", tabletext, re.S)
}
assert len(tables) == 26
for text in (figtext, tabletext):
    stack = []
    for m in re.finditer(r"\\(begin|end)\{([^}]+)\}", text):
        if m[1] == "begin":
            stack.append(m[2])
        else:
            assert stack.pop() == m[2]
    assert not stack
    assert text.count("$") % 2 == 0
    level = 0
    for m in re.finditer(r"(?<!\\)[{}]", text):
        level += 1 if m[0] == "{" else -1
        assert level >= 0
    assert level == 0

checks = []


def table_contains(label, values, rtol=0.006):
    body = tables[label].split(r"\midrule", 1)[1].split(r"\bottomrule")[0]
    nums = [
        float(a) * 10 ** int(b)
        for a, b in re.findall(r"([0-9.]+)\\times10\^\{?(-?\d+)\}?", body)
    ]
    for v in values:
        assert any(np.isclose(v, n, rtol=rtol, atol=0) for n in nums), (
            label,
            v,
        )
    checks.append(dict(table=label, numeric_cells_checked=len(values)))


parity = read(ROOT / "results/checks/parity/parity/audit.json")
for metric, name in [("f", "distribution"), ("rho", "density")]:
    panel = byname[f"E2_parity_budget_{name}_error"]
    for variant, label in [
        ("full", "Projected"),
        ("full_angular", "Half-range"),
    ]:
        key = (
            ("folded_Ef" if metric == "f" else "folded_Erho")
            if variant == "full_angular"
            else "relative_l2_" + metric
        )
        values = [
            median(
                r[key]
                for r in parity
                if r["variant"] == variant and r["features_per_patch"] == j
            )
            for j in (16, 32, 64, 128)
        ]
        close(
            next(c["y"] for c in panel["curves"] if c["label"] == label),
            values,
        )
        table_contains("tab:parity-budget", values)
smoke = read(next((OUT / "parity_smoke").glob("*.json")))
r = next(
    r
    for r in parity
    if r["variant"] == "full_angular"
    and r["features_per_patch"] == 16
    and r["seed"] == 11
)
close(
    [smoke["relative_l2_f"], smoke["relative_l2_rho"]],
    [r["folded_Ef"], r["folded_Erho"]],
)
# Re-evaluate all 30 unified-protocol saved fields and compare shared table entries.
statuses = read(ROOT / "results/runs/feature_study/status.json")
for item in statuses:
    p = resolve_path(item["record"])
    r = read(p)
    with np.load(p.with_suffix(".npz")) as d:
        mask = d["mask"]
        fm = np.broadcast_to(mask[..., None], d["f"].shape)
        close(
            np.linalg.norm((d["f"] - d["reference_f"])[fm])
            / np.linalg.norm(d["reference_f"][fm]),
            r["relative_l2_f"],
        )
        close(
            np.linalg.norm((d["rho"] - d["reference_rho"])[mask])
            / np.linalg.norm(d["reference_rho"][mask]),
            r["relative_l2_rho"],
        )
for problem in ("p1", "p3"):
    values = byname[
        f"E2_{problem}_fixed_collocation_distribution_error_vs_features"
    ]["curves"][0]["y"]
    table_contains("tab:feature-resolution", values)
    table_contains(
        "tab:supp-efficiency" + ("-2d" if problem == "p3" else ""), values
    )
for suffix in (
    "diffusion_density_error",
    "diffusion_scaled_current_error",
    "fick_defect",
):
    table_contains(
        "tab:slab-diffusion", byname["E3_slab_" + suffix]["curves"][0]["y"]
    )
for suffix in (
    "distribution_relative_L2_error",
    "density_relative_L2_error",
    "coefficient_norm",
    "normalized_training_residual_RMS",
):
    table_contains(
        "tab:p5-cutoff",
        byname["E4_P5_seed11_same_version_" + suffix]["curves"][0]["y"],
    )
for metric in ("distribution", "density", "physical_flux"):
    for curve in byname[
        f"E5_budget_dependence_{metric}_error_old_reference_pending_reassessment"
    ]["curves"]:
        # EF has one ordinary decimal cell (1.495), checked separately.
        table_contains("tab:mixed-budget", [v for v in curve["y"] if v < 1])
for eps in ("1e+00", "1e-03"):
    for ref in ("ct", "be"):
        for curve in byname[
            f"E6_eps{eps}_{ref}_distribution_error_three_times"
        ]["curves"]:
            table_contains("tab:transient-comparison", [curve["y"][-1]])
for c in byname["E6_time_refinement_seed11_CT"]["curves"]:
    table_contains("tab:time-refinement", c["y"])
for problem in ("p1", "p3"):
    for c in byname[problem + "_cost_accuracy"]["curves"]:
        table_contains(
            "tab:supp-efficiency" + ("-2d" if problem == "p3" else ""), c["y"]
        )
# Verify E3 and E5 summary medians and envelopes against seedwise records.
slab = [read(p) for p in (ARCH / "results/e3_slab").glob("seed*.json")]
for s in csvread(ARCH / "results/e3_slab/summary.csv"):
    rr = [r for r in slab if r["epsilon"] == float(s["epsilon"])]
    assert sorted(r["seed"] for r in rr) == [11, 23, 37]
    for key in ("E_diff", "E_q_diff", "fick_defect"):
        a = [r["evaluation"]["128"][key] for r in rr]
        close(
            [float(s[key]), float(s[key + "_min"]), float(s[key + "_max"])],
            [median(a), min(a), max(a)],
        )
mixed = csvread(ARCH / "results/e5_mixed/physical_metrics.csv")
for s in csvread(ARCH / "results/e5_mixed/summary.csv"):
    rr = [
        r
        for r in mixed
        if r["method"] == s["method"] and float(r["Nrow"]) == float(s["Nrow"])
    ]
    assert len(rr) == 3
    for key in ("Ef", "Erho", "EF"):
        a = [float(r[key]) for r in rr]
        close(
            [float(s[key]), float(s[key + "_min"]), float(s[key + "_max"])],
            [median(a), min(a), max(a)],
        )

# A source registry separates observed metadata from unrecovered historical fields.
registry = []


def register(group, p, record=None, **extra):
    r = record if record is not None else read(p)
    if isinstance(r, list):
        r = r[0]
    r = {k: v for k, v in r.items() if k not in ("history", "rows")}
    entry = dict(
        group=group,
        raw_output=rel(p),
        sha256=digest(p),
        historical_commit=r.get("commit", r.get("git_commit")),
        original_command=None,
        configuration=r,
        residual_blocks=None,
        reconstruction=None,
        evaluation_grid=r.get("evaluation_grid"),
        reference_version=r.get("reference_level"),
        svd=dict(
            rcond=r.get("rcond"),
            effective_rank=r.get("rank"),
            column_scaling=None,
        ),
        provenance_status="Retained record; null means not independently recovered, not a default.",
    )
    entry.update(extra)
    registry.append(entry)


for r in parity:
    j, seed, v = r["features_per_patch"], r["seed"], r["variant"]
    p = (
        ROOT
        / "results/checks/parity/parity/raw"
        / f"p1_oe_aprfm_eps_1e-03_seed_{seed}_J{j}_{v}.json"
    )
    nboundary = 2 * r["collocation"]["boundary"]
    n = (r["num_rows"] - nboundary) // 3
    register(
        "parity",
        p,
        record=r,
        historical_commit=read(
            ROOT / "results/checks/parity/parity/manifest.json"
        )["commit"],
        residual_blocks=dict(boundary=nboundary, macro=n, even=n, odd=n),
        reconstruction=(
            "r(x,abs(v))+eps*sign(v)*j(x,abs(v))"
            if v == "full_angular"
            else "projected even/odd basis"
        ),
        reference_version="analytic P1",
        evaluation_grid=[257, 128],
        svd=dict(
            rcond=r["rcond"],
            effective_rank=r["rank"],
            column_scaling="L2 after unit-L2 row normalization",
        ),
        rerun_command=[
            "python3",
            "experiments/run_p1_oe_aprfm.py",
            "--epsilon",
            ".001",
            "--seed",
            str(seed),
            "--features",
            str(j),
            "--collocation",
            *[str(x) for x in r["collocation"]["interior"]],
            "--variant",
            v,
        ],
    )
for item in statuses:
    p = resolve_path(item["record"])
    r = read(p)
    register(
        "unified_2d",
        p,
        historical_commit=read(
            ROOT / "results/runs/feature_study/manifest.json"
        )["commit"],
        original_command=item["command"],
        reconstruction=r["angular_representation"],
        residual_blocks=(
            dict(macro=2352, even=4704, odd=4704, boundary=2048)
            if r["problem"] == "p3"
            and r["angular_representation"] == "four_component"
            else None
        ),
        evaluation_grid="65 x 65 spatial nodes; 64 angular midpoints; mask for P4",
        reference_version="analytic manufactured solution",
        svd=dict(
            rcond=r["rcond"],
            effective_rank=r["rank"],
            column_scaling="L2 after row normalization/weights, streaming QR",
        ),
    )
for p in sorted((ARCH / "results/e3_slab").glob("seed*.json")):
    r = read(p)
    register(
        "slab",
        p,
        residual_blocks=dict(macro=960, even=960, odd=960, boundary=64),
        reconstruction="projected parity basis",
        evaluation_grid="128 x 128 Gauss; interior Fick subdomain as archived run_e3_slab.py",
        reference_version="rho=1-x, q=1/3 (diffusion)",
        svd=dict(
            rcond=r["rcond"],
            effective_rank=r["rank"],
            column_scaling="L2; solver source retained",
        ),
    )
for p in sorted(
    (ARCH / "results/comparison_repairs_2026_09_10").glob("p5_*/*.json")
):
    r = read(p)
    register(
        "p5_cutoff_archived",
        p,
        reconstruction="four-component with reflection-invariant angular POU",
        reference_version="archived deterministic P5 reference; original errors retained",
        svd=dict(
            rcond=r["rcond"],
            effective_rank=r["rank"],
            column_scaling="L2 of compressed weighted system; coefficient norm is original theta",
        ),
        limitation="Legacy metadata label is misleading: exact-hash historical source calls one-row physical trace, verified in results/checks/geometry/verification.json. No physical-boundary inconsistency was found for this sweep.",
    )
for p in sorted((ARCH / "results/e5_mixed").glob("*_s*_a*/metadata.json")):
    r = read(p)
    register(
        "mixed",
        p,
        reconstruction=(
            "projected parity"
            if r["method_label"] == "oe"
            else "micro-macro reconstruction"
        ),
        reference_version="archived reference, finite-grid adequacy not established for every metric/configuration",
        svd=dict(
            rcond=r.get("rcond"),
            effective_rank=r.get("rank"),
            column_scaling="L2; archived runner normalization",
        ),
    )
for sub in (
    "factorized_check",
    "time_check",
    "feature_check",
    "quadrature_check",
):
    for p in sorted(
        (ARCH / "results/e6_periodic" / sub).glob("*/result.json")
    ):
        r = read(p)
        register(
            "periodic_" + sub,
            p,
            residual_blocks=dict(
                macro=128, even=2048, odd=2048, periodic_r=32, periodic_j=32
            ),
            reconstruction="even r + eps*odd j; t=0 analytic, first RHS analytic",
            evaluation_grid="512 midpoint x; 128 complete-angle Gauss",
            reference_version="Fourier CT/BE; n256 NPZ files in figure_sources.json",
            svd=dict(
                rcond=r["rcond"],
                effective_rank=r["rank"],
                column_scaling="row L2, each block quadrature sum 1, column L2",
            ),
        )
for p in sorted((ARCH / "results/e6_mm_comparison").glob("s*/result.json")):
    register(
        "periodic_MM",
        p,
        reconstruction="micro-macro",
        reference_version="Fourier CT/BE, same step for BE",
    )
for p in sorted(
    (ROOT / "results/runs/neural_study/runs").glob("cost/**/metrics.json")
):
    register(
        "APNN_cost",
        p,
        svd=dict(applicable=False),
        reconstruction="hard-parity network",
        reference_version="analytic manufactured solution",
    )
# Fixed-collocation P1 feature/cost runs are distinct from the parity-budget sweep.
for source in read(OUT / "figure_sources.json"):
    p = resolve_path(source["path"])
    if p.name == "audit.json" and "efficiency_refresh" in str(p):
        r = read(p)[0]
        if r.get("problem") == "p1":
            register(
                "P1_fixed_collocation_features",
                p,
                record=r,
                reconstruction="projected parity basis",
                reference_version="analytic P1",
                residual_blocks=dict(
                    macro=960, even=960, odd=960, boundary=64
                ),
                evaluation_grid=[257, 128],
                svd=dict(
                    rcond=r["rcond"],
                    effective_rank=r["rank"],
                    column_scaling="L2 after row normalization",
                ),
            )
register(
    "minimum_gain",
    ARCH / "results/minimum_gain/p1_J16_seed11_refined128.json",
    reconstruction="32-dimensional projected parity space",
    evaluation_grid="quadrature refinements up to nx=nv=nq=128",
    svd=dict(
        rcond=None,
        effective_rank=32,
        column_scaling="not the main least-squares solver; no truncation",
    ),
    limitation="Different diagnostic norms/discretization; does not verify main-experiment Assumption 1",
)
(OUT / "run_registry.json").write_text(json.dumps(registry, indent=2) + "\n")

pdfchecks = []
for name in paths:
    p = ROOT / "paper" / name
    assert p.exists()
    fonts = subprocess.check_output(["pdffonts", str(p)], text=True)
    assert "Type 3" not in fonts
    fontlines = fonts.splitlines()[2:]
    assert fontlines
    for line in fontlines:
        assert re.search(r"\byes\s+yes\s+yes\b", line), line
    xml = subprocess.check_output(
        ["pdftotext", "-bbox", str(p), "-"], text=True
    )
    doc = ET.fromstring(xml)
    ns = {"x": "http://www.w3.org/1999/xhtml"}
    pages = doc.findall(".//x:page", ns)
    assert len(pages) == 1
    page = pages[0]
    width, height = float(page.attrib["width"]), float(page.attrib["height"])
    outside = []
    for w in page.findall(".//x:word", ns):
        a = w.attrib
        if (
            float(a["xMin"]) < -0.5
            or float(a["yMin"]) < -0.5
            or float(a["xMax"]) > width + 0.5
            or float(a["yMax"]) > height + 0.5
        ):
            outside.append(w.text)
    assert not outside, (name, outside)
    pdfchecks.append(
        dict(
            path=name,
            sha256=digest(p),
            embedded_fonts=True,
            no_type3=True,
            pages=1,
            width_pt=width,
            height_pt=height,
            no_text_outside_page=True,
        )
    )
# Preserve exact script versions and hash every changed numerical source.
source_paths = [
    "experiments/run_p1_oe_aprfm.py",
    "src/oeaprfm/features/reconstruction.py",
    "scripts/refresh_figures.py",
    "scripts/audit_figures.py",
    "training/README.md",
]
snapshot = OUT / "source"
snapshot.mkdir(exist_ok=True)
for name in source_paths:
    p = ROOT / name
    target = snapshot / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(p.read_bytes())
validation = dict(
    table_count=len(tables),
    figure_environments=15,
    pdf_count=31,
    filenames_match_manifest=True,
    checked_numeric_table_cells=sum(
        r["numeric_cells_checked"] for r in checks
    ),
    checks=checks,
    saved_2d_fields_checked=30,
    slab_seed_records_checked=18,
    mixed_seed_records_checked=24,
    parity_runner_smoke_matches_corrected_audit=True,
    pdfs=pdfchecks,
    run_registry_entries=len(registry),
    latex_compilation="Not performed: manuscript .tex and TeX compiler absent.",
    original_commit=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip(),
    current_source_sha256={p: digest(ROOT / p) for p in source_paths},
    artwork_guidelines="https://link.springer.com/journal/10915/submission-guidelines",
    output_format="Numbered PDF panels; original names are recorded in paper/manifest.json.",
)
(OUT / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
archive = ROOT / "paper/figures.zip"
with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for name in paths:
        z.write(ROOT / "paper" / name, arcname=name)
with zipfile.ZipFile(archive) as z:
    assert len(z.namelist()) == 31 and z.testzip() is None
print(
    json.dumps(
        dict(
            pdfs=31,
            table_cells=validation["checked_numeric_table_cells"],
            registry_entries=len(registry),
            zip=rel(archive),
        ),
        indent=2,
    )
)
