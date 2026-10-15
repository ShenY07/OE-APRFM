# OE-APRFM

奇偶分解渐近保持随机特征方法，用于稳态和时间依赖辐射输运方程。项目使用 Python、JAX/Flax 和 SciPy；OE-APNN 神经网络基线使用可选的 PyTorch。

## 保留的功能

- P1–P5 稳态求解、P6 角度依赖算例，以及 E3/E5/E6 扩展实验。
- OE-APRFM、MM-APRFM、普通 RFM、OE-SI-DSA/Krylov 和 OE-APNN 对比。
- 参考解生成、角域与特征规模比较、物理矩和误差统计。
- 模型保存/加载、训练曲线和密度场绘图。

`models/comparison_index.csv` 对应 30 组 OE-APRFM/OE-APNN 基准模型：P1–P5，ε=1、10⁻³，种子 7/11/17。OE-APNN 另有 18 组训练成本实验；OE-APRFM 另有角域比较模型和 E6 时间轨迹。

**问题定义：** 基准 P5 是与 OE-APNN 相同的方孔制造解。非均匀介质 P5 是独立扩展，不能与方孔结果混用。统一训练入口默认选择方孔；二维实验入口通过 `--p5-definition square_hole` 或 `heterogeneous` 显式区分。

## 目录与入口

```text
src/oeaprfm/       问题配置、随机特征、约束、几何、线性求解和检查点
src/baselines/     oe_apnn、mm_oerfm、oe_si_dsa 的实际实现
experiments/      单次求解、参考解生成和批量实验
training/train.py 统一选择方法与问题
visualization/    保存结果的绘图入口
scripts/          结果统计、物理诊断和资产索引/校验
tests/            数值、模型加载和命令入口测试
data/references/  P1–P5 参考数据及 extensions 扩展参考数据
data/manifest.json 随仓库保存的数据、模型与精选场的 SHA-256 清单
models/           已保存模型、两个加载器和跨方法索引
results/          精选对比结果与本地生成输出
```

`experiments/run_p1_oe_aprfm.py` 支持 P1/P6；`run_p2_oe_aprfm.py` 求解 P2；`run_p3_oe_aprfm.py` 支持 P3/P4/P5。核心二维组装位于 `features/oe_assembly.py`，流式 QR 位于 `solvers/streaming_qr.py`。确定性参考求解器位于 `src/baselines/oe_si_dsa/solver.py`。

## 环境与安装

验证环境为 Linux、Python 3.10，采用 CPU 和双精度。依赖版本列于 `requirements*.txt`。从项目根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
export PYTHONPATH="$PWD/src:$PWD"
export JAX_ENABLE_X64=true
export JAX_PLATFORMS=cpu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
```

OE-APNN 训练、加载及相关测试还需：

```bash
python -m pip install -r requirements-apnn.txt
```

无需外部服务、密钥或环境配置文件。单次神经网络训练支持 CPU；`experiments/retrain_oe_apnn.py` 是需要 CUDA 的完整批量训练入口。

## 快速运行

小规模 OE-APRFM 求解与绘图：

```bash
python training/train.py --method oe_aprfm --problem p1 \
  --epsilon 0.001 --seed 11 --features 16 --collocation 8 8 \
  --output-dir results/smoke/oe_aprfm
python visualization/plot_results.py \
  results/smoke/oe_aprfm/p1_oe_aprfm_eps_1e-03_seed_11.npz \
  --output results/smoke/density.png
```

输出包含预测场、JSON 指标及同名 `*_model/` 检查点目录。这个小规模示例用于验证运行，不是精度基准。

其他基线：

```bash
python training/train.py --method mm_aprfm --problem p1 \
  --epsilon 1 --seed 11 --features 16 --collocation 8 8 \
  --output-dir results/smoke/mm_aprfm
python -m baselines.oe_si_dsa.run --problem p1 --epsilon 1 \
  --grid-1d 16 8 --output-dir results/smoke/oe_si_dsa
python training/train.py --method oe_apnn --problem p1 --steps 1 \
  --interior-samples 16 --boundary-samples 16 --quadrature-points 4 \
  --output-dir results/smoke/oe_apnn
```

使用 `python training/train.py --method oe_aprfm --problem p3 --runner-help` 查看参数。MM 稳态入口支持 P1–P4/P6；OE-APNN 支持 P1–P5。

## 数据与模型

P1/P3/P4/P5 使用解析制造解，`data/references/p*/exact.npz` 保存评价网格及解。P2 使用 `data/references/p2/` 的 Gauss–Legendre 数值参考解。扩展实验数据集中在 `data/references/extensions/`；训练配点由求解器在线生成。

```bash
python -m experiments.exact_references
python -m experiments.generate_reference --problem p2 --epsilon 0.001 --level B
```

生成数据会更新文件；确认新资产后执行 `python scripts/index_assets.py` 重建校验清单。索引命令还会验证 30 对基准模型是否齐全。

加载已有 OE-APRFM 模型：

```python
from models.load_aprfm import load_model
predict = load_model("models/oe_aprfm/baseline/p1/eps_1e-03/seed_11")
f = predict([0.2, 0.8], [0.3, -0.3])
```

二维模型传入 `x, y, theta`。加载器核对随机特征参数并恢复双精度预测。OE-APNN 加载器是 `models/load_apnn.py`，返回求解器与 `r/j` 网络；调用 `solver.reconstruct(nets, x, v)` 或二维的 `solver.reconstruct(nets, x, y, theta)`。

`python -m experiments.retrain_oe_aprfm --workers 2` 生成完整的 30 组基准模型，并核验保存场与重载预测。完整实验比快速示例耗时长。已有模型和单次运行结果不应被视为跨硬件的速度比较。

## 扩展与报告

- `experiments/run_e3_slab.py`、`run_e5_mixed.py`：扩散极限与混合介质实验。
- `experiments/run_e6_periodic.py`、`reference_e6.py`：周期输运、隐式后向 Euler 及独立 Fourier 参考；MM 时间扩展位于 `baselines.mm_oerfm.periodic`。
- `experiments/run_angle_campaign.py`：角域对比；`scripts/report_angle_domains.py` 校验配对系统并输出汇总。
- `scripts/refresh_2d_tables.py`：P3/P4 特征规模及表示方式实验。
- `scripts/report_e3_slab.py`、`report_e5_mixed.py`、`report_apnn_retrain.py` 和 `finalize_e6.py`：汇总各自已完成的运行。

报告脚本需要先运行对应实验，不能仅靠模型文件生成完整研究报告。部分本地生成结果不会提交；代码不依赖论文、旧路径映射或外部历史目录。

## 验证

本项目无独立编译构建步骤或类型检查配置。

```bash
python -m compileall -q src experiments scripts tests training visualization models
python scripts/check_assets.py
python -m pytest -q tests --ignore=tests/test_apnn.py
# 安装可选 PyTorch 后：
python -m pytest -q tests
```

自动化测试覆盖配置、奇偶组装、PoU、流式 QR、物理矩、模型重载和基线入口。GitHub Actions 执行 CPU 语法、资产与非神经网络测试。

数值参考的迭代收敛不等于离散精度充分；结论仍需网格、角度和时间步加密验证。第三方基线的作者归属与许可说明见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
