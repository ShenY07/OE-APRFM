#!/usr/bin/env python3
"""Build the unified baseline-result table from committed result files."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def read(path):
    return json.loads(path.read_text())


def fmt(value):
    return "—" if value is None else f"{float(value):.3e}"


def deterministic(method, data, sample="seed 11", note=""):
    return {
        "method": method,
        "sample": sample,
        "ef": fmt(data.get("relative_l2_f")),
        "er": fmt(data.get("relative_l2_rho")),
        "time": fmt(data.get("total_seconds", data.get("runtime_seconds"))),
        "status": ("收敛" if data.get("converged") is True else
                   "未收敛" if data.get("converged") is False else "完成"),
        "note": note,
    }


def apnn(problem, label):
    by_seed = {}
    for root in (RESULTS / "baselines/oe_apnn", RESULTS / "baselines/oe_apnn_imported"):
        for path in (root / problem / label).glob("seed_*/metrics.json"):
            record = read(path)
            if record.get("evaluation_status") == "complete":
                by_seed[int(record["seed"])] = record
    records = list(by_seed.values())
    if len(records) != 3:
        raise RuntimeError(f"{problem}/{label}: expected seeds 7,11,17, found {sorted(by_seed)}")
    ef = np.array([r["E_f"] for r in records], float)
    er = np.array([r["E_rho"] for r in records], float)
    tm = np.array([r["train_time_s"] for r in records], float)
    value = lambda x: f"{x.mean():.3e} ± {x.std(ddof=1):.3e}"
    return {"method": "OE-APNN", "sample": "3 seeds", "ef": value(ef),
            "er": value(er), "time": value(tm), "status": "完成",
            "note": "方孔制造解版" if problem == "P5" else ""}


def oe_path(problem, epsilon):
    tag = f"{epsilon:.0e}"
    if problem == "P1":
        return next((RESULTS / "domain_decomposition/1d_fixed_local").glob(
            f"p1_oe_aprfm_eps_{tag}_seed_11_*p11_f64.json"))
    if problem == "P2":
        return RESULTS / "consistency/p2" / f"p2_oe_aprfm_eps_{tag}_seed_11_fixed.json"
    return next((RESULTS / "quadrant/final").glob(
        f"{problem.lower()}_oe_aprfm_eps_{tag}_seed_11_*.json"))


def main():
    rows, missing = [], []
    labels = {1.0: "eps_1e0", 1e-3: "eps_1e-3"}
    for pi in range(1, 6):
        problem = f"P{pi}"
        for epsilon in (1.0, 1e-3):
            current = []
            path = oe_path(problem, epsilon)
            current.append(deterministic("OE-APRFM", read(path)))

            mm_path = RESULTS / "baselines/mm_aprfm" / f"p{pi}_mm_aprfm_eps_{epsilon:.0e}_seed_11.json"
            if mm_path.exists():
                current.append(deterministic("MM-APRFM", read(mm_path)))
            else:
                missing.append(("MM-APRFM", problem, epsilon, "无正式结果"))

            current.append(apnn(problem, labels[epsilon]))

            suffix = "" if pi <= 2 else "_krylov"
            si_path = RESULTS / "baselines/oe_si_dsa" / f"p{pi}_oe_si_dsa{suffix}_eps_{epsilon:.0e}.json"
            if si_path.exists():
                current.append(deterministic("OE-SI-DSA", read(si_path), sample="确定性"))
            else:
                missing.append(("OE-SI-DSA", problem, epsilon, "无正式结果"))

            if problem == "P2":
                with (RESULTS / "tables/iterative_solver_summary.csv").open() as handle:
                    item = next(r for r in csv.DictReader(handle)
                                if np.isclose(float(r["epsilon"]), epsilon))
                current.append({"method": "SI-DSA", "sample": "确定性", "ef": "参考解",
                                "er": "参考解", "time": fmt(item["runtime_seconds"]),
                                "status": "收敛" if item["converged"] == "true" else "未收敛",
                                "note": "P2 数值参考解生成器"})
            if problem == "P1":
                rfm_path = RESULTS / "baselines/rfm" / f"p1_rfm_eps_{epsilon:.0e}_seed_11.json"
                if rfm_path.exists():
                    current.append(deterministic("RFM", read(rfm_path)))
                else:
                    missing.append(("RFM", problem, epsilon, "代表性失败/退化实验尚未运行"))
                si_path = RESULTS / "baselines/si_dsa" / f"p1_si_dsa_eps_{epsilon:.0e}.json"
                if si_path.exists():
                    current.append(deterministic("SI-DSA", read(si_path), sample="确定性"))
                else:
                    missing.append(("SI-DSA", problem, epsilon, "代表性收敛实验尚未运行"))
            for entry in current:
                rows.append((problem, epsilon, entry))

    output = RESULTS / "tables/baseline_model_comparison.md"
    lines = ["# 基准模型结果对比", "",
             "| 问题 | ε | 方法 | 统计口径 | E_f | E_ρ | 时间/s | 状态 | 备注 |",
             "|---|---:|---|---|---:|---:|---:|---|---|"]
    for problem, epsilon, entry in rows:
        lines.append(f"| {problem} | {epsilon:.0e} | {entry['method']} | {entry['sample']} | "
                     f"{entry['ef']} | {entry['er']} | {entry['time']} | {entry['status']} | {entry['note']} |")
    lines += ["", "## 尚缺结果", "",
              "| 方法 | 问题 | ε | 缺失内容 |", "|---|---|---:|---|"]
    for method, problem, epsilon, reason in missing:
        lines.append(f"| {method} | {problem} | {epsilon:.0e} | {reason} |")
    lines += ["| OE-APNN | P5 | 1e+00, 1e-03 | 当前异质介质 P5 尚无同问题模型；方孔版不可直接横向比较 |"]
    lines += ["", "注：OE-APNN 时间为 GPU 训练时间；随机特征方法时间为特征生成、组装、求解和评估总时间；"
              "OE-SI-DSA/SI-DSA 时间为迭代求解时间。不同时间口径已保留，不直接合并为加速比。",
              "", "注：OE-APNN P5 是归档协议的中心方孔制造解；其他方法 P5 是当前异质介质版本。"]
    output.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(rows)} result rows and {len(missing)} missing rows to {output}")


if __name__ == "__main__":
    main()
