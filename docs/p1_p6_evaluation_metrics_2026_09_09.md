# P1--P6 按四项评判标准整理的现有结果

最新填写汇总：见 [E1–E5 已填写表格](/mnt/store2/sheny-y23/OE-APRFM/docs/e1_e5_filled_tables_2026_09_09.md)。E5 已完成24/24组，P5相邻截断两组已完成；下方较早的运行中状态仅为历史记录。参考、求积、计时及P5同版本基线缺口按新表列示，不视为全部验收通过。

日期：2026-09-09

## P1 截断必要性补充诊断（8组已完成）

| rcond | OE Ef | OE rank/128 | OE 系数范数 | MM Ef | MM rank/128 | MM 系数范数 |
|---|---:|---:|---:|---:|---:|---:|
| 1e-6 | 2.33520e-6 | 79 | 18.27 | 2.06491e-7 | 46 | 4.22 |
| 1e-8 | 7.92951e-7 | 114 | 70.20 | 5.07454e-9 | 66 | 5.50 |
| 1e-10 | 5.17723e-7 | 127 | 405.61 | 6.34914e-10 | 79 | 6.25 |
| 1e-12 | 5.20906e-7 | 128 | 466.61 | 4.02202e-11 | 81 | 7.05 |

本次固定空间内矩阵及右端哈希完全一致。结果不支持 rcond=1e-6 强截断的必要性：OE 放宽后误差降低，1e-10后近似平台，满秩也未发生误差爆炸；MM 放宽后精度持续提高，但到1e-12仍截断了47个方向，故不能声称 MM 完全无需截断。OE系数范数明显增大不等于已证明扰动不稳定。本轮未做受控扰动或低于1e-12扫描。OE为2944行，MM为2828行，本表用于各自阈值敏感性而非等行数效率比较。新运行结果独立归档，不覆盖旧冻结精度数字。

执行 `scripts/run_p1_cutoff_check.py`：固定 epsilon=1e-3、seed 11、单分区、OE 每场64特征、MM macro/micro各64特征，均128系数；保留原 U(0,1) 初始化。扫描 rcond=1e-6/1e-8/1e-10/1e-12，不改变各自配点。两方法统一行归一化、列均衡及 gelsd；因此此 MM 数字不直接等同于旧归一化配置的冻结结果。每种方法校验矩阵/RHS哈希不随阈值变化。

输出目录 `results/p1_cutoff_check/`；每组 `diagnostic.json` 保存误差、秩、系数范数与矩阵哈希，全部结束后生成 `summary.csv`。残差单位分别为 OE 原始残差和 MM 行归一化残差，只在方法内部随阈值比较。该单种子单 epsilon 检查不能直接证明 P2/P5/E5 或所有随机空间必须强截断。

## 最新执行要求：E1--E5（替代下文旧阶段闸门）

本轮采用用户最新五组实验要求；旧的“只有 OE 获胜才进入 mixed-scale”闸门作废。E5 所有预设档位均报告，不以 OE/MM 胜负筛选。P2/P4/P6 为补充；不新增含时、lattice 或更多几何。

| 组别 | 当前执行状态 |
|---|---|
| E1 固定空间稳定性 | 已完成外层 64×64→128×128、内层 64→128 分别加密；32/32 满秩，无截断。见下方交付文件 |
| E2 制造解与消融 | 复用已有档案；尚需按新要求逐项整理来源和图表 |
| E3 无源 slab 扩散极限 | 18 组已完成，128 系数、2944 行、rcond=1e-12；独立输运参考及评价求积加密已检查，见下方结果 |
| E4 P5 二维异质 | 物理函数已从 src/configuration/p5_heterogeneous_2d.py 核对：sigma_s=1+y，sigma_a=0.1，Q=0；左右入流 1+0.2y，下侧 1，上侧 1.2。最终运行覆盖配置为分区 1×1×2、J=128、scale=0.25、rcond=1e-6、block_weights=[10,1,1,1]；配置文件默认分区不能替代运行档案 |
| E5 外部混合尺度 | 使用用户规定 kappa(x)，全局 epsilon=1、sigma_s=1/kappa；1152 总系数、24 组角预算比较正在执行；两级参考已生成，尚待完整验收 |

初始化已按用户“按新要求运行”冻结：E3/E5 新计算采用 U(-1,1)；E1 加密延续原固定空间，E2/E4 复用数据保留原初始化标记。

### E3 新计算结果

18 组（6 epsilon × 3 seeds）已完成；固定 128 系数、2944 行、角平均8点、rcond=1e-12。误差使用独立64/128阶物理求积；流直接由 j 计算。以下为三种子中位数。

| epsilon | E_diff | E_q_diff | Fick 缺陷 [0.1,0.9] |
|---|---:|---:|---:|
| 1 | 0.253823 | 0.576451 | 0.0720124 |
| 0.1 | 0.0568511 | 0.118173 | 0.00394378 |
| 0.01 | 0.00636775 | 0.0129080 | 0.000843284 |
| 0.001 | 0.000642376 | 0.00130225 | 0.0000863355 |
| 1e-4 | 0.0000642801 | 0.000130316 | 0.00000862315 |
| 1e-6 | 6.28653e-7 | 1.27129e-6 | 1.01248e-7 |

独立 diamond-difference/GMRES 输运参考采用 256×64→512×128 加密。epsilon=1 的参考 f/rho 差异约 1.01e-3 / 1.75e-4，epsilon=0.1 为 1.29e-4 / 2.28e-5；六组有限 epsilon 方法误差均通过“加密差异低于方法误差10%”检查。该检查包含不同角网格的插值差异，是数值分辨率检查。

原始数据与完整范围：`results/e3_slab/summary.csv`、`seed*.json`；图为 `diffusion_limit.pdf` 和 `profiles.pdf`。执行脚本：`run_e3_slab.py`、`check_e3_transport.py`、`report_e3_slab.py`（均位于 scripts）。

E5 已通过 `scripts/run_e5_mixed.py` 启动预设的24组比较，两级参考位于 `results/e5_mixed/reference`。尚未完成全部档位和角平均加密验收，不能标为最终交付。每组完成后写 metadata.json，可断点续跑。

E1 交付：`results/minimum_gain/p1_J16_seed11_refined128.json`、`quadrature_check.csv`、`beta_epsilon.png`、`beta_epsilon.pdf`；图表脚本为 `scripts/report_minimum_gain.py`。内外求积变化均远低于 1% 项目标准；此为数值收敛检查，不是连续 epsilon 区间认证。

## 主文执行定位（少改现有实验版）

现有 P1--P6 不再共同承担“OE 全面优于 MM”的论证，而按下表分工。新增实验只回答：parity 是否改善有限 RF 表示、折叠角域是否节省角向信息、以及经典欠分辨多尺度问题上是否保持正确 AP 行为。

新增实验的第一优先级不是继续比较原始或截断后的矩阵条件数，而是计算与稳定性定理一致的有限维诊断

\[
\beta_{\varepsilon,m}
=\inf_{0\ne U_m}\frac{\|\mathscr T_\varepsilon U_m\|_Y}{\|U_m\|_X},
\qquad
\beta_{\varepsilon,m}^2=\lambda_{\min}(K_\varepsilon,G),
\]

其中 `Gij=(Ψi,Ψj)X`，`Kε,ij=(TεΨi,TεΨj)Y`。该量用于区分函数空间中的稳定性和随机基系数表示的病态性；`κeff(A)` 降为补充诊断。

| 问题 | 主文角色 | 保留指标 | 是否新增实验 |
|---|---|---|---|
| P1 | parity-conforming representation efficiency，核心机制实验 | parity/unconstrained 的 off-grid `Ef--J` 三种子曲线 | 不新增大规模计算；用现有数据生成 Fig. A |
| P2 | 1D heterogeneous validation | `Ef`、`Eρ`、reference adequacy | 原则上不补 |
| P3 | smooth 2D / dimensional extension | `Ef`、`Eρ` | 原则上不补；仅在二维角域折叠零成本接入时扩展 |
| P4 | geometry robustness | `Ef`、`Eρ`、带孔域可用性 | 不做 MM 优势比较 |
| P5 | non-manufactured heterogeneous boundary-driven validation | `Ef`、`Eρ`；若现有数据链容易则补 `q` | 不做大规模调参 |
| P6 | representation trade-off / limitation case | 主文只保留简短 OE/MM 对照；`Eg`、`Eq`、even/odd 诊断移至补充材料 | 不再承担 OE 优势证明 |

### 当前比较范围（取代旧阶段闸门）

E2 负责 parity/unconstrained 消融；E5 只比较 OE-folded 与 MM-full，共24组，不新增第三方法或根据胜负筛选档位。取消原 two-material/lattice 扩展路线。半角域不自动节省一半未知量；报告实际残差行数、对称性/角平均造成的重复行及数值秩，三者不得混淆。

### 第一优先级：有限维 inf--sup / 最小增益诊断

只在 P1 的小型固定 RF 空间上实施，建议固定 `J=16` 或 `J=32`、seed 11，并扫描 `ε=1,1e-1,1e-2,1e-3,1e-6`；只有理论算子在 `ε=0` 有明确定义且无需用含 `1/ε` 的制造 source 才加入 `ε=0`。

执行要求：

1. 从理论中逐项抄录 `X` 范数、`Y` 范数、边界迹项、分量的 ε 权重和 `Tε` 的齐次算子定义；source/RHS 不进入 `Kε`。
2. `G` 与 `Kε` 使用独立于训练配点的 tensor-product 加密求积组装；至少两个逐级加密层次，并报告 `β` 的相邻层次相对变化。
3. 对加权状态评价矩阵做带列置换的薄 QR：`CX P=Q R`；三角求解得到 `(CY P) R^{-1}`，取其最小奇异值。避免形成 Gram 矩阵放大病态性。
4. 保存 `CX` 的数值秩、秩判断阈值、最大/最小奇异值、`β`、`λmin=β²` 和求积层次。本诊断不删除小方向；若数值满秩无法确认，报告未解析。删除非零方向后只能称受限子空间诊断，不能称原空间的 β。
5. 主结论只写成“给定随机有限维空间和数值求积下的稳定性诊断”；不将其表述为连续空间稳定常数的严格下界或认证。

用户已提供精确定义：X 为未缩放的 `(r,j)` 空间 H1 乘积范数；Y 为三个内部残差与无通量权重的入流 L2 范数，四块系数为 1。折叠后内部权重为 `wx wv`、宏观块为 `wx`、每侧边界为 `wv/2`，左右符号分别为 `r+εj` 和 `r-εj`。诊断只用齐次算子，不调用制造 source，因此 ε=0 直接代入。

已实现 `scripts/diagnose_oe_minimum_gain.py`：复用实际 `RandomFeatureSpaceXV` 的单分区 parity 空间，空间导数经自动微分作用于完整特征函数。默认 J=16/场、32 个总系数、seed=11、scale=1、sigma_s=1、sigma_a=0。独立投影求积与外层范数求积分别加密。解析校验 `span{(x,0)}` 给出 `sqrt(5/8)`；另独立推导 `span{(1,0),(0,v)}`，验证左右边界交叉项抵消、β=1。

首轮结果（64×64 外层求积、64 点角平均，完整 32 维空间无截断）：

| ε | β（约） |
|---|---:|
| 1 | 0.16840205 |
| 0.1 | 0.04933323 |
| 0.01 | 0.03802870 |
| 0.001 | 0.03739774 |
| 1e-6 | 0.03733276 |
| 0 | 0.03733270 |

这些值仅对应固定空间与六个 ε 采样点，不能将样本最小值称为整个 `[0,1]` 上的严格下界。完整求积与秩数据见 `results/minimum_gain/p1_J16_seed11.json`。复现：`OPENBLAS_NUM_THREADS=1 python3 scripts/diagnose_oe_minimum_gain.py`。

### AP 补充实验的最小范围

只保留一组固定预算、非制造的稳态 diffusion-limit 对照。若能为 P5 明确并实现正确的极限边界条件，就复用 P5；否则采用更简单的标准 slab。输出限定为 `Ediff`、必要的 current/Fick defect、固定预算和 reference refinement，不因此增加含时或 lattice 实验。

## 口径说明

- 本文只整理仓库中已经保存的 OE-APRFM / MM-APRFM 结果，不补造未运行的实验。
- 除注明为 `seed 11` 外，中心值均为 seeds 11/23/37 的中位数；`[min,max]` 为三种子范围。
- `κeff=σmax/σmin,eff`、`reff` 与实际 SVD 截断后的求解矩阵一致。`Cemp` 只是误差/离散残差的经验敏感度代理，不等于受控右端扰动实验。
- `off-grid` 指在独立于训练配点的固定评估网格上计算误差；它不等于连续角域上的严格误差界。
- P1、P3、P4、P6 有解析制造解；P2、P5 相对于通过 refinement 检查的确定性输运参考解计算误差。

## 一、有限维离散下的稳定性

| 问题 | OE-APRFM：奇异值、秩与条件性 | OE-APRFM：精度/敏感度 | MM-APRFM 现有证据 | 结论 |
|---|---|---|---|---|
| P1：1D 制造解 | 128 系数；`reff=128/128`。ε=1：σmax=8.83、σmin,eff=1.12e-8、κeff=7.88e8；ε=1e-3：8.95、7.81e-10、1.15e10（奇异值均为 seed 11） | 冻结表 median `Ef`: 2.71e-8 → 1.21e-8 → 1.21e-8（ε=1,1e-3,1e-6）；median `Cemp`: 8.34 → 4.82 → 4.86 | seed 11、128 系数：ε=1e-3 时 `reff=81/128`、κeff=3.81e11、Ef=4.94e-11；三 ε 均只有 81 个有效秩 | 解精度对 ε 稳定，但两种方法都不能称为矩阵良态；MM 在该 SVD 截断设置下取得高精度；尚不能据此断言强截断必要 |
| P2：1D 非均匀介质 | 1024 系数；ε=1 / 1e-3 的 median `reff=673/1024` / `490/1024`，κeff=9.92e5 / 9.86e5；σmin,eff≈1.00e-5 / 1.32e-5 | 冻结表 median `Ef`=2.41e-2 / 2.08e-3，`Eρ`=3.34e-3 / 1.10e-3；当前只有 ε=1、1e-3 的认证参考 | 未归档同口径三种子奇异值/扰动表 | 条件数由 `rcond=1e-6` 控制，但有效秩损失明显；不能据此声称未截断系统稳定 |
| P3：2D 光滑制造解 | 四分量、512 系数，`reff=512/512`；median κeff=7.99e5 / 7.19e6，σmin,eff=1.44e-5 / 1.61e-6（ε=1 / 1e-3） | median `Ef`=3.90e-3 / 2.01e-3，`Eρ`=1.33e-3 / 1.03e-3；`Cemp`=6.77 / 2.74 | ε=1e-3 的匹配效率档案含三种子结果，但未形成跨 ε 奇异值表 | 满秩且小 ε 精度未退化；条件性约恶化一个数量级 |
| P4：带圆孔 2D 制造解 | 四分量、512 系数，`reff=512/512`；median κeff=1.25e6 / 6.91e6，σmin,eff=9.04e-6 / 1.66e-6 | median `Ef`=1.56e-2 / 1.63e-2，`Eρ`=1.04e-2 / 1.38e-2；`Cemp`=8.40 / 9.81 | 未归档同口径 MM 结果 | 两尺度误差同量级且满秩，但小 ε 的密度误差略升、条件性变差 |
| P5：2D 光滑非均匀、边界驱动 | 1024 系数；已归档样本仅 `reff≈321--363/1024`、κeff≈(9.63--9.92)e5，σmin,eff≈(1.21--1.30)e-5 | median `Ef`=4.23e-2 / 3.62e-2，`Eρ`=4.76e-3 / 4.60e-3；三 seed 的 Ef 范围很窄（ε=1: 4.21e-2--4.27e-2；ε=1e-3: 3.55e-2--3.64e-2） | 未归档同口径 MM 结果 | seed 稳健性较好，但这是截断后的稳定结果，不能解释为满秩稳定 |
| P6：非零角向制造解 | 128 系数、`reff=128/128`；median κeff=7.09e8 → 1.04e10 → 1.04e10（ε=1,1e-3,1e-6） | median `Ef`=2.17e-5 → 1.64e-5 → 1.64e-5；`Cemp`=3.48 → 4.72 → 4.72 | 等系数预算下三种子 `Ef`=1.58e-5 → 4.82e-7 → 4.85e-7；MM 奇异值摘要未单独归档 | 两种方法的解误差均未随 ε 缩小而爆炸；OE 仍是高条件数满秩系统 |

**稳定性总判断：** 现有证据支持“截断后的固定有限维求解在测试 ε 范围内保持可用精度”，不支持“离散算子一致良态”。仓库尚无受控 `δA/δb` 扰动后直接测量 `||δc||` 或解误差变化的实验。

## 二、角向结构精确嵌入、稀疏采样、off-grid 与 seed robustness

| 问题 | 精确嵌入的角向结构 | 稀疏预算 / off-grid / seed 结果 | 结论与缺口 |
|---|---|---|---|
| P1 | `r(x,-v)=r(x,v)`、`j(x,-v)=-j(x,v)` 由表示层精确满足 | ε=1e-3、独立 257×128 endpoint grid、相同 Ncoef/Nrow：J=16 时 parity `Ef=1.12e-2 [9.53e-4,2.05e-2]`，unconstrained `4.27e-1 [3.41e-2,4.34e-1]`；J=32 时 `6.81e-5 [1.17e-5,1.09e-4]` 对 `7.97e-2 [1.83e-2,4.30e-1]`；J=64 时 `1.21e-8 [5.90e-9,5.21e-7]` 对 `1.42e-2 [2.91e-3,5.58e-2]` | 唯一完成“同预算 parity/unconstrained + off-grid + 三 seed”闭环的算例；强证据不只停留在结构正确 |
| P2 | 1D 奇偶表示精确满足 parity | 正式误差在独立参考/评估网格上计算；三 seed median `Ef`=2.41e-2 / 2.08e-3，但无角特征预算消融 | 支持整体 off-grid 与 seed 可重复性；不支持“稀疏角采样优势”的独立归因 |
| P3 | 第一象限四分量 `(j1,r1,j2,r2)` 精确生成全角域的 even/odd 结构 | 三 seed、独立测试网格；median `Ef`=3.90e-3 / 2.01e-3；无同预算 unconstrained 四分量对照，也无独立角节点稀疏扫描 | 结构与 seed 结果已验证，稀疏角向鲁棒性尚未验证 |
| P4 | 与 P3 相同的四分量象限表示，适配带孔区域 | 三 seed、独立测试网格；`Ef` 范围：ε=1 为 1.53e-2--2.01e-2，ε=1e-3 为 1.61e-2--1.84e-2 | 几何改变后 seed 范围仍有限；缺 unconstrained/off-grid 角向专项消融 |
| P5 | 四分量 parity 表示；正式冻结配置含角向分区 | 三 seed 对确定性参考解：ε=1 `Ef=4.23e-2 [4.21e-2,4.27e-2]`，ε=1e-3 `3.62e-2 [3.55e-2,3.64e-2]` | seed robustness 很强，但不能把它单独归因于角向结构；缺相同预算无约束对照 |
| P6 | 非零角向制造解显式检验 even/odd 重构；OE 与 MM 都使用结构化表示 | 128 系数、三 seed、独立 257×128 网格；OE `Ef` 范围在 ε=1e-6 为 1.04e-5--2.38e-5，MM 为 1.78e-7--7.88e-7 | 证明结构化表示能处理非零角向项；但真解仅含各向同性 even 部与线性 `v` 微观项，不是一般高阶角各向异性基准 |

**角向结构总判断：** P1 给出最强的因果消融证据；P2--P5 主要是“结构正确且整体误差可复现”；P6 补上了非零角向物理量，但角复杂度仍低。现有“稀疏”扫描改变的是随机特征数 J，不是角求积节点数，因此不应写成任意稀疏角求积鲁棒。

## 三、奇偶矩物理解耦：flux/current 与 Fick-law defect

这里的 scaled flux/current 定义为 `q=<v f>/ε`。P1、P3 的真解各向同性，真 flux 和真 microscopic field 均为零，因此只能报告绝对误差；P6 的真 flux 非零，可以报告相对误差。

| 问题 | 已有 flux/current 与奇偶矩结果 | OE 与 MM 的物理解读 | Fick-law defect 状态 |
|---|---|---|---|
| P1 | ε=1e-3，OE J=128：flux absolute L2 median=1.61e-12，`g_even`=8.61e-10，`g_odd`=1.84e-11；MM J=128：3.48e-11、6.32e-14、6.03e-11 | 两者都把本例应为零的奇部/流控制到很小；P1 不能检验非零 current 的相对精度 | 未计算 |
| P2 | 未保存独立 flux、`g_even/g_odd` 诊断 | 仅有 `Ef/Eρ`，不能推出奇偶矩物理解耦 | 未计算 |
| P3 | ε=1e-3，OE J=128：flux absolute L2 median=4.95e-3，`g_even`=3.80，`g_odd`=1.53e-2；MM J=128：2.22e-7、1.04e-8、3.15e-7 | 在该固定预算档案中 MM 对零 flux/微观场的控制明显优于 OE；OE 的密度误差不能代表微观矩误差 | 未计算 |
| P4 | 未保存独立 flux、`g_even/g_odd` 诊断 | 带孔几何下只有 `Ef/Eρ`，不足以判断 current 解耦 | 未计算 |
| P5 | 未保存独立 flux、`g_even/g_odd` 诊断 | 边界驱动本应是有价值的 current 测试，但当前档案不含该量 | 未计算 |
| P6 | OE relative scaled-flux error：6.28e-5 → 1.62e-4 → 1.61e-4；MM：6.82e-6 → 3.34e-6 → 3.47e-6。ε=1e-6 时 OE `g_even` absolute error=5.14、`g_odd`=1.06e-4；MM 分别 6.44e-6、6.02e-6 | OE 的 odd-g/flux 随 ε 稳定，但 `(f-ρ)/ε` 放大了 even-g 缺陷；MM 在本等系数预算制造例上实现了更完整的宏微/奇偶解耦 | 未计算；scaled-flux error 不能改名为 Fick-law defect |

**物理解耦总判断：** P6 是目前唯一同时具有非零真 flux、三种子和 OE/MM 等系数预算比较的算例。它支持 MM-APRFM 在该制造例上的解耦优势，也暴露了 OE-APRFM 中“宏观密度和 flux 准确，但完整 microscopic field 未必准确”的现象。

## 四、固定粗预算下的 diffusion compatibility

| 问题 | 固定预算随 ε 缩小的现有结果 | 是否可认证 diffusion limit？ | 需要补充的关键实验 |
|---|---|---|---|
| P1 | OE Ncoef=128：median `Eρ`=2.64e-9 → 4.73e-9 → 4.78e-9；MM seed 11 的 `Ef`=5.13e-11 → 4.94e-11 → 9.27e-11 | 否。制造 source 含 ε 依赖，且真 flux 为零 | 固定空间/角/特征预算，对照独立 diffusion PDE；报告 `||ρε-ρdiff||` 与 Fick defect |
| P2 | OE 固定 2×4 分区、1024 系数：median `Eρ`=3.34e-3 → 1.10e-3（ε=1 → 1e-3） | 否。ε=1e-6 参考未认证，且没有 diffusion PDE 对照 | 先补可认证的小 ε kinetic/reference，再做同边界层处理的 diffusion 对照与 ε 斜率 |
| P3 | OE 四分量 512 系数：median `Eρ`=1.33e-3 → 1.03e-3 | 否。制造 source 随 ε 变化 | 固定粗网格/角向/特征预算，保存 kinetic 与 diffusion 两套密度及 current/Fick defect |
| P4 | OE 四分量 512 系数：median `Eρ`=1.04e-2 → 1.38e-2 | 否。只说明误差未数量级爆炸，且复杂几何的 diffusion 边界条件尚未对照 | 明确孔边界的 diffusion 极限条件，分离内部误差与边界层误差 |
| P5 | OE 固定 1024 系数：median `Eρ`=4.76e-3 → 4.60e-3 | 否。与 AP 相容，但没有 `ρdiff` 或 Fick-law 数据 | 这是最适合补做的非制造 thick-diffusion 案例：固定粗预算、同介质和入流，报告密度极限、current、边界层外误差 |
| P6 | OE Ncoef=128：`Eρ`=5.90e-6 → 1.59e-5 → 1.59e-5，flux error=6.28e-5 → 1.62e-4 → 1.61e-4；MM：`Eρ`=3.43e-6 → 4.82e-7 → 4.85e-7，flux error=6.82e-6 → 3.34e-6 → 3.47e-6 | 否。强烈支持固定预算 ε-robustness，尤其是 MM，但仍是 ε-dependent 制造解而非独立 diffusion-limit convergence | 用 ε 无关物理数据生成 kinetic family，比较独立 diffusion 解并报告收敛斜率 |

**diffusion compatibility 总判断：** P1--P6 的固定预算误差总体与 diffusion compatibility 相容，其中 P5 最适合承载经典 thick-diffusion 故事，P6 最适合展示宏微量与 current 的稳定性。但当前任何一项都不能单独作为严格 AP/diffusion-limit 认证。

## E1–E5 当前综合结论（优先于旧 P1–P6 汇总）

| 证据 | 当前可支持的结论 | 限制 |
|---|---|---|
| E1 | 完整32维固定空间，无截断，理论范数最小增益的内外求积检查通过 | 不推广到所有RF空间或连续epsilon区间 |
| E2 | P1同预算parity消融支持有限表示效率；制造解用于精度与结构验证 | 需逐项审计冻结配置，不替代扩散极限 |
| E3 | 非制造、固定物理数据slab已给出密度/流扩散极限趋近及Fick缺陷；epsilon=1、0.1另有输运参考 | 组装角平均敏感性及独立入流/流平衡正在收尾；不要求P2–P6各补一套扩散实验 |
| E4 | 最终P5验证二维异质边界驱动问题；实际精度属于截断求解结果 | 相邻rcond及独立残差审计未完成 |
| E5 | 按预设24组比较同一物理问题下OE/MM的角预算响应 | 参考、角求积、计时尚需验收，不预先声称角向效率优势 |

E1的函数范数诊断、E2的配点检查与实际截断求解是不同证据；采样最小二乘仍需要离散—连续残差范数等价，不能合并为无条件离散准最优性结论。P6的微观恢复局限继续保留。

## 旧 P1–P6 分级结论（仅对应旧档案，不代表 E1–E5 整体）

| 评判标准 | 当前证据等级 | 最稳妥的表述 |
|---|---|---|
| 有限维离散稳定性 | 中等 | 截断后的固定有限维系统在测试 ε 范围内保持误差稳定；矩阵本身可高度病态，尚无直接扰动实验 |
| 角向结构精确嵌入 | P1 强，P2--P6 中等 | parity 约束被精确嵌入；P1 的同预算 off-grid 三种子消融显示它显著提升低特征预算精度和最坏 seed |
| 奇偶矩物理解耦 | P6 强，其余不足 | P6 上 MM 的 flux 和完整 microscopic field 均稳定；OE 的 flux 稳定但 even-g 缺陷在小 ε 下被放大 |
| 粗分辨率 diffusion compatibility | 初步 | 固定预算误差没有随 ε→0 爆炸，与 diffusion compatibility 相容；尚未完成独立 diffusion PDE/Fick-law 验证 |

## 数据来源

- 稳定性与精度：`results/tables/table_2.csv`、`table_3.csv`、`table_4.csv`、`table_7.csv`、`table_8.csv`；最终三种子中心值优先采用 `results/requirement_2026_08_20/tables_frozen/` 下的冻结表。
- P2/P4/P5 冻结三种子汇总：`results/requirement_2026_08_20/tables_frozen/table_5_extended_examples.csv`。
- P1 parity 稀疏特征预算：`results/requirement_2026_08_20/tables_frozen/table_S4_parity_feature_budget.csv`。
- flux 与 microscopic parity：`results/requirement_2026_08_20/transport_diagnostics/summary.csv`。
- P6 OE/MM 等系数预算：`results/p6_mm_comparison/summary.csv` 与 `results/p6_mm_comparison/REPORT.md`。
- diffusion-limit 边界说明：`scripts/build_publication_tables.py` 的 “Diffusion-limit verification 状态”。
