"""Local global-scaled MM+BE extension; six prescribed E6 comparisons only."""

import json, csv, time, hashlib
from pathlib import Path
import numpy as np
from scipy.linalg import svd
from run_e6_periodic import gauss

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/e6_mm_comparison"


class Space:
    def __init__(self, seed):
        rng = np.random.default_rng(seed)
        self.m = rng.uniform(-1, 1, (128, 2))
        self.g = rng.uniform(-1, 1, (128, 3))

    def macro(self, x):
        a = np.tanh(
            (2 * np.asarray(x)[..., None] - 1) * self.m[:, 0] + self.m[:, 1]
        )
        return a, 2 * self.m[:, 0] * (1 - a * a)

    def micro(self, x, v):
        x, v = np.broadcast_arrays(x, v)
        a = np.tanh(
            (2 * x[..., None] - 1) * self.g[:, 0]
            + v[..., None] * self.g[:, 1]
            + self.g[:, 2]
        )
        return a, 2 * self.g[:, 0] * (1 - a * a)


def run(seed, eps):
    dt = 0.002
    out = OUT / f"s{seed}_eps{eps:.0e}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "result.json").exists():
        return
    start = time.perf_counter()
    sp = Space(seed)
    x = (np.arange(128) + 0.5) / 128
    vp, wp = gauss(16, True)
    v = np.r_[-vp[::-1], vp]
    w = np.r_[wp[::-1], wp] / 2
    xx, vv = np.meshgrid(x, v, indexing="ij")
    G, Gx = sp.micro(xx, vv)
    M, Mx = sp.macro(x)
    q, wq = gauss(64)
    xq, vq = np.meshgrid(x, q, indexing="ij")
    Gq, Gqx = sp.micro(xq, vq)
    mean = np.einsum("xvk,v->xk", Gq, wq)
    fluxx = np.einsum("xvk,v->xk", Gqx, wq * q)
    vb, wb = gauss(32, True)
    vb = np.r_[-vb[::-1], vb]
    wb = np.r_[wb[::-1], wb] / 2
    gl, _ = sp.micro(0, vb)
    gr, _ = sp.micro(1, vb)
    ml, _ = sp.macro(0)
    mr, _ = sp.macro(1)
    A = np.concatenate(
        (
            np.c_[M / dt, fluxx],
            np.concatenate(
                (
                    vv[..., None] * Mx[:, None, :],
                    (1 + eps**2 / dt) * G
                    + eps * (vv[..., None] * Gx - fluxx[:, None, :]),
                ),
                axis=-1,
            ).reshape(-1, 256),
            np.c_[np.zeros_like(mean), mean],
            np.r_[ml - mr, np.zeros(128)][None, :],
            np.c_[np.zeros_like(gl), gl - gr],
        )
    )
    H = np.concatenate(
        (
            np.c_[M / dt, np.zeros_like(M)],
            np.concatenate(
                (np.zeros_like(G), eps**2 / dt * G), axis=-1
            ).reshape(-1, 256),
            np.zeros((193, 256)),
        )
    )
    assert A.shape == (4417, 256) and H.shape == A.shape
    weights = np.r_[
        np.ones(128) / 128, np.tile(w / 128, 128), np.ones(128) / 128, 1.0, wb
    ]
    for a, b in [
        (0, 128),
        (128, 4224),
        (4224, 4352),
        (4352, 4353),
        (4353, 4417),
    ]:
        np.testing.assert_allclose(weights[a:b].sum(), 1)
    norms = np.linalg.norm(A, axis=1)
    factor = np.sqrt(weights) / np.where(norms > 1e-30, norms, 1)
    Aw = A * factor[:, None]
    cs = np.linalg.norm(Aw, axis=0)
    cs = np.where(cs > 1e-14, cs, 1)
    u, s, vt = svd(Aw / cs, full_matrices=False)
    keep = s > 1e-12 * s[0]

    def solve(b):
        return (vt[keep].T @ ((u[:, keep].T @ (factor * b)) / s[keep])) / cs

    assembly = time.perf_counter() - start
    xe = (np.arange(512) + 0.5) / 512
    ve, we = gauss(128)
    ME, _ = sp.macro(xe)
    xx, vv = np.meshgrid(xe, ve, indexing="ij")
    GE, _ = sp.micro(xx, vv)
    cc = solve(np.r_[np.ones(128) / dt, np.zeros(4289)])
    constant = dict(
        macro_RMS=float(np.sqrt(np.mean((ME @ cc[:128] - 1) ** 2))),
        micro_RMS=float(np.sqrt(np.mean((GE @ cc[128:]) ** 2))),
    )
    assert max(constant.values()) < 1e-8, constant
    rhs = np.r_[(1 + 0.2 * np.cos(2 * np.pi * x)) / dt, np.zeros(4289)]
    history = []
    saved = dict(
        x=xe, v=ve, w=we, macro_parameters=sp.m, micro_parameters=sp.g
    )
    c = None
    adv = 0
    for step in range(101):
        if step == 0:
            macro = 1 + 0.2 * np.cos(2 * np.pi * xe)
            g = np.zeros((512, 128))
        else:
            tic = time.perf_counter()
            c = solve(rhs if step == 1 else H @ c)
            adv += time.perf_counter() - tic
            macro = ME @ c[:128]
            g = GE @ c[128:]
        f = macro[:, None] + eps * g
        rho = f @ we
        current = g @ (ve * we)
        d0 = float(np.sqrt(np.mean((g @ we) ** 2)))
        history.append(
            dict(
                t=step * dt,
                mass_drift=float(abs(rho.mean() - 1)),
                D0=d0,
                min_f=float(f.min()),
            )
        )
        if step in (0, 10, 50, 100):
            saved.update(
                {
                    f"f_{step}": f,
                    f"rho_{step}": rho,
                    f"rho_macro_{step}": macro,
                    f"q_{step}": current,
                    f"g_{step}": g,
                }
            )
            if c is not None:
                saved[f"coefficients_{step}"] = c.copy()
    np.savez_compressed(out / "fields.npz", **saved)
    record = dict(
        method="local global-scaled MM-APRFM+BE",
        epsilon=eps,
        dt=dt,
        seed=seed,
        Ncoef=256,
        Nrow=4417,
        rank=int(keep.sum()),
        macro_features=128,
        micro_features=128,
        rcond=1e-12,
        constant_regression=constant,
        feature_hash=hashlib.sha256(
            sp.m.tobytes() + sp.g.tobytes()
        ).hexdigest(),
        assembly_seconds=assembly,
        advance_seconds=adv,
        total_seconds=time.perf_counter() - start,
        history=history,
    )
    (out / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    print(seed, eps, constant, history[-1], flush=True)


def report():
    rows = []
    for p in sorted(OUT.glob("s*/result.json")):
        r = json.loads(p.read_text())
        d = np.load(p.parent / "fields.npz")
        x = d["x"]
        wv = d["w"]
        eps = r["epsilon"]
        ref = np.load(
            ROOT
            / f"results/e6_periodic/references/eps{eps:.0e}_dt2e-03_n256.npz"
        )
        v, w = gauss(256)
        for step in (10, 50, 100):
            t = step * 0.002
            phase = np.exp(2j * np.pi * x)
            for kind in ("ct", "be"):
                a = ref[f"{kind}_{t:.2f}"]
                fr = 1 + 0.2 * np.real(phase[:, None] * a[256:])
                rho = 1 + 0.2 * np.real(phase * (a[:256] @ w))
                q = 0.2 * np.real(phase * (a[:256] @ (v * w))) / eps

                def err(z, b, weight=1):
                    return float(
                        np.sqrt(
                            np.sum(weight * (z - b) ** 2)
                            / np.sum(weight * b * b)
                        )
                    )

                rows.append(
                    dict(
                        method="MM",
                        epsilon=eps,
                        seed=r["seed"],
                        t=t,
                        reference=kind,
                        Ef=err(d[f"f_{step}"], fr, wv),
                        Erho=err(d[f"rho_{step}"], rho),
                        Eq=err(d[f"q_{step}"], q),
                        Erho_macro=err(d[f"rho_macro_{step}"], rho),
                        D0=r["history"][step]["D0"],
                        rank=r["rank"],
                        Nrow=r["Nrow"],
                        max_mass_drift=max(
                            h["mass_drift"] for h in r["history"]
                        ),
                    )
                )
    with (ROOT / "results/e6_periodic/reference_comparison.csv").open() as f:
        for r in csv.DictReader(f):
            if (
                not r["run"].startswith("factorized_check/")
                or float(r["epsilon"]) not in (1, 0.001)
                or float(r["t"]) == 0
            ):
                continue
            meta = json.loads(
                (
                    ROOT / "results/e6_periodic" / r["run"] / "result.json"
                ).read_text()
            )
            rows.append(
                dict(
                    method="OE",
                    epsilon=float(r["epsilon"]),
                    seed=int(r["seed"]),
                    t=float(r["t"]),
                    reference=r["reference"],
                    Ef=float(r["Ef"]),
                    Erho=float(r["Erho"]),
                    Eq=float(r["Eq_relative"]),
                    Erho_macro="",
                    D0="",
                    rank=meta["rank"],
                    Nrow=4288,
                    max_mass_drift=max(
                        h["mass_drift"] for h in meta["history"]
                    ),
                )
            )
    with (OUT / "seedwise.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = []
    for eps in (1, 0.001):
        for method in ("OE", "MM"):
            row = dict(epsilon=eps, method=method)
            for ref, key, label in [
                ("ct", "Ef", "Ef_CT"),
                ("be", "Ef", "Ef_BE"),
                ("be", "Erho", "Erho_BE"),
                ("be", "Eq", "Eq_BE"),
            ]:
                values = [
                    r[key]
                    for r in rows
                    if r["method"] == method
                    and r["epsilon"] == eps
                    and r["t"] == 0.2
                    and r["reference"] == ref
                ]
                assert len(values) == 3
                row[label] = float(np.median(values))
                row[label + "_min"] = min(values)
                row[label + "_max"] = max(values)
            summary.append(row)
    with (OUT / "summary.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    lines = [
        "# E6：OE/MM 含时定量对照\n",
        "新增6条MM轨迹，复用6条OE轨迹；现有两张E6主图不变。主表为t=0.2三seed中位数，完整范围见summary.csv，逐seed/时刻见seedwise.csv。\n",
        "| epsilon | Method | Ef CT | Ef BE | Erho BE | Eq BE |",
        "|---|---|---|---|---|---|",
    ]
    for r in summary:
        lines.append(
            "| "
            + " | ".join(
                str(r[k]) if k == "method" else f"{r[k]:.6g}"
                for k in [
                    "epsilon",
                    "method",
                    "Ef_CT",
                    "Ef_BE",
                    "Erho_BE",
                    "Eq_BE",
                ]
            )
            + " |"
        )
    lines += [
        "\n## 实现与比较边界\n",
        "MM为本地全局缩放f=rho+epsilon*g的含时扩展。宏观/微观各128特征，总256；单分区，128空间中点，16正Gauss角点及其镜像共32内部角点，64完整角求积。宏观与零均值各128行，微观4096行，宏观周期1行、微观周期64行，总4417行。\n",
        "五块为宏观时间平衡、微观时间平衡、零均值、宏观周期、微观周期；逐行归一化后各块积分权重和为1，再列均衡及rcond=1e-12 SVD。宏观rho/dt+<vg_x>，微观(1+epsilon²/dt)g+v rho_x+epsilon(vg_x-<vg_x>)；右端分别旧rho/dt与epsilon²旧g/dt。第一步解析初值，后续复用SVD因子，不预合成推进矩阵。\n",
        "宏观tanh[w_x(2x-1)+b]、微观tanh[w_x(2x-1)+w_v v+b]，同seed顺序生成U(-1,1)参数并保存。不是OE的单因素parity消融；名义预算相同不等于表达空间相同。\n",
        "密度由重构f积分，流直接由<vg>计算。D0和独立宏观密度误差另存。常数单步两场RMS均需小于1e-8，否则中止；初始相对流误差不报告。CT和BE同时保留，不按胜负删档；没有隔离计时，不声称加速倍数。\n",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines))
    print(summary, flush=True)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for seed in (11, 23, 37):
        for eps in (1, 0.001):
            run(seed, eps)
    report()
