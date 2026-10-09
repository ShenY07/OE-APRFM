"""Finish E6 using saved coefficients only: grids, acceptance, tables, figures."""

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/e6_final_mpl")
import csv, json
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from e6_protocol import validate

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/e6_periodic"


def save(name, rows):
    with (OUT / name).open("w") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def read(name):
    with (OUT / name).open() as f:
        return list(csv.DictReader(f))


def relative(a, b, w=1):
    return float(np.sqrt(np.sum(w * (a - b) ** 2) / np.sum(w * b * b)))


def evaluate(params, c, x, v):
    J = len(params)
    fs = []
    js = []
    for start in range(0, len(x), 32):
        xx = x[start : start + 32, None, None]
        vv = v[None, :, None]
        p = np.tanh(
            (2 * xx - 1) * params[:, 0] + vv * params[:, 1] + params[:, 2]
        )
        m = np.tanh(
            (2 * xx - 1) * params[:, 0] - vv * params[:, 1] + params[:, 2]
        )
        js.append(((p - m) / 2) @ c[:J])
        fs.append(((p + m) / 2) @ c[J:])
    return np.concatenate(fs), np.concatenate(js)


def metrics(r, d, nx, nv, reference):
    x = (np.arange(nx) + 0.5) / nx
    v, w = np.polynomial.legendre.leggauss(nv)
    w = w / 2
    ref = np.load(
        OUT / f"references/eps{r['epsilon']:.0e}_dt{r['dt']:.0e}_n256.npz"
    )
    vn, wn = np.polynomial.legendre.leggauss(256)
    wn = wn / 2
    result = []
    for t in (0.02, 0.1, 0.2):
        c = d[f"coefficients_{round(t/r['dt'])}"]
        rr, jj = evaluate(d["parameters"], c, x, v)
        f = rr + r["epsilon"] * jj
        rho = rr @ w
        q = jj @ (w * v)
        phase = np.exp(2j * np.pi * x)
        for kind in ("ct", "be"):
            a = ref[f"{kind}_{t:.2f}"]
            av = a[:256] if nv == 256 else a[256:]
            fr = 1 + 0.2 * np.real(phase[:, None] * av)
            dr = 1 + 0.2 * np.real(phase * (a[:256] @ wn))
            qr = 0.2 * np.real(phase * (a[:256] @ (vn * wn))) / r["epsilon"]
            result.append(
                dict(
                    t=t,
                    reference=kind,
                    nx=nx,
                    nv=nv,
                    Ef=relative(f, fr, w),
                    Erho=relative(rho, dr),
                    Eq_absolute=float(np.sqrt(np.mean((q - qr) ** 2))),
                    Ediff_BE=relative(
                        rho,
                        1
                        + 0.2
                        * (1 + 4 * np.pi**2 * r["dt"] / 3)
                        ** (-round(t / r["dt"]))
                        * np.cos(2 * np.pi * x),
                    ),
                )
            )
    return result


def main():
    batches = {
        "factorized_check",
        "time_check",
        "feature_check",
        "quadrature_check",
    }
    paths = [
        p
        for p in sorted(OUT.glob("*/*/result.json"))
        if p.parent.parent.name in batches and validate(p)
    ]
    if len(paths) != 20:
        raise RuntimeError(
            f"Expected 20 prescribed E6 trajectories, found {len(paths)}; see docs/reproduction.md"
        )
    records = {
        str(p.parent.relative_to(OUT)): json.loads(p.read_text())
        for p in paths
    }
    refined = []
    checks = []
    checkpoint = OUT / "test_refinement_checkpoint.json"
    completed = (
        json.loads(checkpoint.read_text()) if checkpoint.exists() else {}
    )
    for path in paths:
        key = str(path.parent.relative_to(OUT))
        r = records[key]
        if key not in completed:
            with np.load(path.parent / "fields.npz") as d:
                completed[key] = sum(
                    [
                        metrics(r, d, nx, nv, None)
                        for nx, nv in ((512, 128), (1024, 128), (512, 256))
                    ],
                    [],
                )
            checkpoint.write_text(json.dumps(completed) + "\n")
        values = completed[key]
        refined.extend([dict(run=key, **a) for a in values])
        for t in (0.02, 0.1, 0.2):
            for kind in ("ct", "be"):
                base = next(
                    a
                    for a in values
                    if a["t"] == t
                    and a["reference"] == kind
                    and a["nx"] == 512
                    and a["nv"] == 128
                )
                for direction, nx, nv in [
                    ("space", 1024, 128),
                    ("angle", 512, 256),
                ]:
                    other = next(
                        a
                        for a in values
                        if a["t"] == t
                        and a["reference"] == kind
                        and a["nx"] == nx
                        and a["nv"] == nv
                    )
                    for metric in ("Ef", "Erho", "Eq_absolute", "Ediff_BE"):
                        change = abs(other[metric] - base[metric])
                        ratio = change / max(abs(other[metric]), 1e-300)
                        checks.append(
                            dict(
                                run=key,
                                t=t,
                                reference=kind,
                                direction=direction,
                                metric=metric,
                                base=base[metric],
                                refined=other[metric],
                                absolute_change=change,
                                relative_change=ratio,
                                pass_one_percent=ratio < 0.01,
                            )
                        )
        print(key, "grid checks complete", flush=True)
    save("test_grid_metrics.csv", refined)
    save("test_grid_checks.csv", checks)
    acceptance = []
    refrows = [
        r for r in read("reference_comparison.csv") if r["run"] in records
    ]
    for r in refrows:
        for metric, delta in [
            ("Ef", "reference_delta_f"),
            ("Erho", "reference_delta_rho"),
            ("Eq_absolute", "reference_delta_q_absolute"),
        ]:
            error = float(r[metric])
            diff = float(r[delta])
            initial = float(r["t"]) == 0
            acceptance.append(
                dict(
                    run=r["run"],
                    t=r["t"],
                    reference=r["reference"],
                    metric=metric,
                    error=error,
                    reference_delta=diff,
                    ratio=diff / error if error > 0 else "",
                    status=(
                        "initial exact state; no relative acceptance"
                        if initial
                        else (
                            "pass"
                            if diff < 0.1 * error
                            else "reference-limited"
                        )
                    ),
                )
            )
    save("reference_acceptance.csv", acceptance)
    diagnostics = []
    for key, r in records.items():
        diagnostics.append(
            dict(
                run=key,
                epsilon=r["epsilon"],
                dt=r["dt"],
                seed=r["seed"],
                J=r["J"],
                nq=r["nq"],
                rank=r["rank"],
                max_mass_drift=max(a["mass_drift"] for a in r["history"]),
                max_periodic_r=max(a["periodic_r"] for a in r["history"]),
                max_periodic_j=max(a["periodic_j"] for a in r["history"]),
                min_f=min(a["min_f"] for a in r["history"]),
                constant_r_error=r["constant_step_regression"][
                    "constant_r_error"
                ],
                constant_j_error=r["constant_step_regression"][
                    "constant_j_error"
                ],
                constant_pass_1e8=max(r["constant_step_regression"].values())
                < 1e-8,
                assembly_factorization_seconds=r[
                    "assembly_factorization_seconds"
                ],
                advance_seconds=r["advance_seconds"],
                total_seconds=r["total_seconds"],
            )
        )
    save("final_trajectory_diagnostics.csv", diagnostics)
    paired = []
    for key, r in records.items():
        if not key.startswith(("feature_check/", "quadrature_check/")):
            continue
        basekey = next(
            k
            for k, b in records.items()
            if k.startswith("factorized_check/")
            and b["seed"] == r["seed"]
            and b["epsilon"] == r["epsilon"]
        )
        with (
            np.load(OUT / key / "fields.npz") as d,
            np.load(OUT / basekey / "fields.npz") as b,
        ):
            for t in (0.02, 0.1, 0.2):
                step = round(t / r["dt"])
                paired.append(
                    dict(
                        run=key,
                        baseline=basekey,
                        t=t,
                        rank=r["rank"],
                        baseline_rank=records[basekey]["rank"],
                        f_change=relative(
                            d[f"f_{step}"], b[f"f_{step}"], d["w"]
                        ),
                        rho_change=relative(
                            d[f"rho_{step}"], b[f"rho_{step}"]
                        ),
                        q_change=relative(d[f"q_{step}"], b[f"q_{step}"]),
                    )
                )
    save("assembly_feature_checks.csv", paired)
    # Main-table medians/ranges keep continuous and BE references separate.
    summary = []
    for eps in (1, 0.1, 0.01, 0.001):
        for t in (0.02, 0.1, 0.2):
            for kind in ("ct", "be"):
                group = [
                    a
                    for a in refrows
                    if a["run"].startswith("factorized_check/")
                    and float(a["epsilon"]) == eps
                    and float(a["t"]) == t
                    and a["reference"] == kind
                ]
                assert len(group) == 3
                row = dict(epsilon=eps, t=t, reference=kind, seeds=3)
                for metric in ("Ef", "Erho", "Eq_relative", "Ediff_BE"):
                    v = [float(a[metric]) for a in group]
                    row[metric] = float(np.median(v))
                    row[metric + "_min"] = min(v)
                    row[metric + "_max"] = max(v)
                summary.append(row)
    save("accuracy_summary.csv", summary)
    time_rows = [
        r
        for r in refrows
        if float(r["t"]) == 0.2
        and (
            r["run"].startswith("time_check/")
            or (
                r["run"].startswith("factorized_check/s11_")
                and float(r["epsilon"]) in (1, 0.001)
            )
        )
    ]
    save("time_refinement.csv", time_rows)
    fig, axes = plt.subplots(3, 3, figsize=(12, 9))
    for eps in (1, 0.1, 0.01, 0.001):
        key = next(
            k
            for k, r in records.items()
            if k.startswith("factorized_check/s11_") and r["epsilon"] == eps
        )
        r = records[key]
        d = np.load(OUT / key / "fields.npz")
        x = d["x"]
        ref = np.load(OUT / f"references/eps{eps:.0e}_dt2e-03_n256.npz")
        vn, wn = np.polynomial.legendre.leggauss(256)
        wn /= 2
        for col, t in enumerate((0.02, 0.1, 0.2)):
            step = round(t / 0.002)
            a = ref[f"be_{t:.2f}"][:256]
            phase = np.exp(2j * np.pi * x)
            line = axes[0, col].plot(
                x, d[f"rho_{step}"], label=f"eps={eps:g}"
            )[0]
            axes[0, col].plot(
                x,
                1 + 0.2 * np.real(phase * (a @ wn)),
                "--",
                color=line.get_color(),
                lw=0.8,
            )
            line = axes[1, col].plot(x, d[f"q_{step}"])[0]
            axes[1, col].plot(
                x,
                0.2 * np.real(phase * (a @ (vn * wn))) / eps,
                "--",
                color=line.get_color(),
                lw=0.8,
            )
            axes[0, col].set_title(f"t={t:g}")
            axes[0, col].set_ylabel("rho")
            axes[1, col].set_ylabel("q")
        h = r["history"]
        axes[2, 0].plot(
            [a["t"] for a in h],
            [a["amplitude"] for a in h],
            label=f"eps={eps:g}",
        )
    tt = np.linspace(0, 0.2, 101)
    axes[2, 0].plot(
        tt,
        0.2 * np.exp(-4 * np.pi**2 * tt / 3),
        "k--",
        label="continuous diffusion",
    )
    axes[2, 0].plot(
        tt,
        0.2 * (1 + 4 * np.pi**2 * 0.002 / 3) ** (-np.arange(101)),
        "k:",
        label="BE diffusion",
    )
    axes[2, 0].set(xlabel="t", ylabel="cosine amplitude")
    axes[2, 0].legend(fontsize=7)
    axes[2, 1].axis("off")
    axes[2, 2].axis("off")
    axes[2, 1].text(
        0,
        0.7,
        "Solid: RF (seed 11)\nDashed profiles: same-dt BE transport\nInitial q=0; no uniform initial-layer claim",
        fontsize=10,
    )
    for ax in axes[:2].ravel():
        ax.set_xlabel("x")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"evolution.{ext}", dpi=180)
    fig, ax = plt.subplots(figsize=(6, 4))
    for t in (0.02, 0.1, 0.2):
        g = sorted(
            [r for r in summary if r["t"] == t and r["reference"] == "be"],
            key=lambda a: a["epsilon"],
        )
        x = [r["epsilon"] for r in g]
        ax.loglog(x, [r["Ediff_BE"] for r in g], "o-", label=f"t={t:g}")
        ax.fill_between(
            x,
            [r["Ediff_BE_min"] for r in g],
            [r["Ediff_BE_max"] for r in g],
            alpha=0.15,
        )
    ax.set(xlabel="epsilon", ylabel="rho difference from BE diffusion")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"discrete_diffusion.{ext}", dpi=180)
    report(
        records, summary, diagnostics, checks, acceptance, paired, time_rows
    )


def report(
    records, summary, diagnostics, checks, acceptance, paired, time_rows
):
    text = [
        "# E6：周期含时输运——最终计算与验收报告\n",
        "20条预设RF轨迹及所需后处理已完成；无新增轨迹。完成检查不表示每个量都通过阈值，以下保留所有失败/分辨率受限项。\n",
    ]

    def table(title, rows, keys):
        text.extend(
            [
                "## " + title + "\n",
                "| " + " | ".join(keys) + " |",
                "|" + "|".join("---" for _ in keys) + "|",
            ]
        )
        for r in rows:
            text.append(
                "| "
                + " | ".join(
                    f"{r[k]:.6g}" if isinstance(r[k], float) else str(r[k])
                    for k in keys
                )
                + " |"
            )
        text.append("")

    failures = [r for r in checks if not r["pass_one_percent"]]
    limited = [r for r in acceptance if r["status"] == "reference-limited"]
    table(
        "完成与验收状态",
        [
            dict(item="有效轨迹", result="20/20，另排除2条作废实现诊断"),
            dict(item="嵌套特征", result="256参数前128行与128参数逐项一致"),
            dict(
                item="常数单步检查",
                result=f"{sum(r['constant_pass_1e8'] for r in diagnostics)}/20 满足两场绝对RMS<1e-8（收尾检查阈值）",
            ),
            dict(
                item="测试加密",
                result=f"{len(checks)-len(failures)}/{len(checks)} 满足指标变化<1%；不通过{len(failures)}项均保留",
            ),
            dict(
                item="参考充分性",
                result=f"{len(limited)}项差异不小于方法误差10%，其余非初始项通过；初始相对判定不适用",
            ),
        ],
        ["item", "result"],
    )
    text.append(
        "测试加密分别采用(512,128)→(1024,128)和(512,128)→(512,256)，覆盖20条轨迹的三个非零剖面时刻。接近舍入误差时相对变化敏感；未自动按绝对阈值改判通过。完整原始/加密值见test_grid_checks.csv。参考f使用无反馈探针ODE共同角点评价，不做角插值。\n"
    )
    table(
        "主精度表（三seed中位数；ct连续输运，be同dt输运）",
        summary,
        ["epsilon", "t", "reference", "Ef", "Erho", "Eq_relative", "Ediff_BE"],
    )
    table(
        "全轨迹诊断和有效秩",
        diagnostics,
        [
            "run",
            "rank",
            "max_mass_drift",
            "max_periodic_r",
            "max_periodic_j",
            "min_f",
            "constant_r_error",
            "constant_j_error",
        ],
    )
    table(
        "时间步加密（seed11，末时刻）",
        time_rows,
        [
            "epsilon",
            "dt",
            "reference",
            "Ef",
            "Erho",
            "Eq_relative",
            "Ediff_BE",
        ],
    )
    table(
        "组装求积与特征加密（同网格场变化）",
        paired,
        [
            "run",
            "t",
            "rank",
            "baseline_rank",
            "f_change",
            "rho_change",
            "q_change",
        ],
    )
    text.extend(
        [
            "## 冻结问题、实现与解释边界\n",
            "epsilon² f_t+epsilon v f_x=<f>-f；x∈(0,1)、v∈[-1,1]、t∈[0,.2]；初值1+.2cos(2πx)，周期边界。无源纯散射，解析r初值和j=0直接进入第一步右端。主dt=.002；epsilon=1/.1/.01/.001，seeds11/23/37。\n",
            "单分区每场J=128，共256系数；tanh[w_x(2x-1)+w_v v+b]，numpy default_rng(seed)的U(-1,1)，r/j共享原始基。主内部128空间中点×16正半域Gauss点，完整64点组装角平均；宏观128行，even/odd各2048行，两个周期块各32行，共4288行。检查轨迹仅改变dt、组装求积或J。\n",
            "五块RM、RE、RO、Br、Bj先按原A行L2归一化，再各自施加和为1的求积权重平方根，列均衡后rcond=1e-12。只分解一次，每步显式更新右端并应用SVD因子；不预合成系数推进矩阵、不重置质量、不重抽特征。旧main两条valid_for_e6=false，所有汇总使用protocol/config/feature校验。\n",
            "连续时间输运误差包含时间离散；同dt BE输运误差衡量RF与残差/代数误差；BE扩散差异包含有限epsilon渐近差异。三者不能互换。初始q=0，不计算其相对误差，不宣称解析小epsilon初始流层。采样最小值不等于严格保正，小质量漂移不等于严格守恒。\n",
            "不在接近RF/代数平台的时间表上强制拟合收敛阶。计时见final_trajectory_diagnostics.csv：组装/分解与推进分列，总耗时含评价，未作为隔离性能比较。\n",
            "主图：[演化与幅值](evolution.pdf)、[离散扩散渐近](discrete_diffusion.pdf)。完整CSV位于results/e6_periodic：accuracy_summary（含min/max）、time_refinement、assembly_feature_checks、test_grid_metrics、test_grid_checks、reference_acceptance、final_trajectory_diagnostics。\n",
            "复现后处理：`OPENBLAS_NUM_THREADS=1 python3 scripts/finalize_e6.py`。不会触发RF求解。E6扩展此前E1–E5的稳态范围，不扩展到其他含时算例。",
        ]
    )
    (OUT / "REPORT.md").write_text("\n".join(text))
    (OUT / "completion.json").write_text(
        json.dumps(
            dict(
                trajectories=len(records),
                test_checks=len(checks),
                test_failed=len(failures),
                reference_limited=len(limited),
                postprocessing_complete=True,
            ),
            indent=2,
        )
        + "\n"
    )
    print(
        "E6 postprocessing complete:",
        len(failures),
        "grid flags;",
        len(limited),
        "reference-limited entries",
        flush=True,
    )


if __name__ == "__main__":
    main()
