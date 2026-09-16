"""Populate closeout tables from archived measurements, without new solves."""
import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
LATEST_NOTICE = '\n> 最新修复批次19/19项计算已完成；P5三个同版本阈值和MM测试加密结果见 [E1–E6当前汇总](experiments_current_summary_2026_09_10.md)。本文件以下旧E4两阈值表保留为历史，不与新批次拼接；E5新参考尚待后处理。\n'
def rows(path):
    with (ROOT/path).open() as f:return list(csv.DictReader(f))
def fmt(x):
    if isinstance(x,bool):return '通过' if x else '未通过'
    try:return f'{float(x):.7g}'
    except (ValueError,TypeError):return str(x)
out=['# E1–E5 已填写结果与验收对应表\n',
     '本文件由 `scripts/build_closeout_tables.py` 从归档结果生成；数据完成与科学验收分开标记。误差为相对量，注明绝对量的除外。旧批次保留原初始化；E3/E5 为 U(-1,1)。\n']
def table(title,data,columns):
    out.append('## '+title+'\n')
    out.append('| '+' | '.join(label for key,label in columns)+' |')
    out.append('|'+'|'.join('---' for _ in columns)+'|')
    for r in data:out.append('| '+' | '.join(fmt(r.get(k,'未归档')) for k,_ in columns)+' |')
    out.append('')
e1=rows('results/minimum_gain/quadrature_check.csv')
d=json.loads((ROOT/'results/minimum_gain/p1_J16_seed11_refined128.json').read_text())
for r in e1:
    q=next(q for q in d['rows'] if q['nx']==128 and q['nv']==128 and q['nq']==128 and q['epsilon']==float(r['epsilon']))
    r['rank_tolerance']=q['rank_tolerance']
table('E1：完整固定空间可靠性',e1,[('epsilon','epsilon'),('beta','beta'),('rank','秩/32'),('rank_tolerance','状态秩阈值'),('inner_relative_change','内层相对变化'),('outer_relative_change','外层相对变化')])
out.append('来源：`results/minimum_gain/quadrature_check.csv` 与 refined128 JSON。所有行无截断；最大内/外相对变化分别约1.93e-14/1.75e-14。仅为固定空间、离散求积及六个epsilon点的诊断。\n')
table('E2：parity预算（epsilon=1e-3；三seed）',rows('results/requirement_2026_08_20/tables_frozen/table_S4_parity_feature_budget.csv'),[(k,k) for k in ['representation','J','Ncoef','Nrow','median_E_f','min_E_f','max_E_f','median_E_rho']])
table('E2：固定配点的特征扫描',rows('results/requirement_2026_08_20/tables_frozen/table_3_feature_resolution_full.csv'),[(k,k) for k in ['problem','representation','features_per_patch','num_columns','num_rows','rcond','relative_l2_f','relative_l2_rho','rank_fraction']])
table('E2：固定特征的P1配点扫描',[r for r in rows('results/requirement_2026_08_20/tables_frozen/table_3_collocation_refinement.csv') if r['problem']=='p1'],[(k,k) for k in ['oversampling_ratio','seeds','relative_l2_f','relative_l2_rho']])
raw=list((ROOT/'results/requirement_2026_08_20/parity_budget_raw').glob('*.json'))
records=[json.loads(p.read_text()) for p in raw]
out.append('E2来源：上述同名冻结CSV，parity逐组原始数据为 `results/requirement_2026_08_20/parity_budget_raw/`。特征扫描与配点扫描分开；parity表仅同J内Ncoef/Nrow匹配，跨J行数变化。\n')
out.append('parity原始档案核对：'+str(len(records))+' 组；rcond='+str(sorted({r.get('rcond') for r in records},key=str))+'；seed='+str(sorted({r.get('seed') for r in records},key=str))+'。同一runner的variant路径复用基础RF生成类；未保存初始化参数哈希的旧档案不能声称已逐参数认证一致。\n')
table('E3：有限epsilon输运误差与参考验收',rows('results/e3_slab/transport_check.csv'),[(k,k) for k in ['epsilon','seed','E_f','E_rho','refinement_f','refinement_rho','ratio_f','ratio_rho']])
out.append('ratio=参考加密差异/方法误差，六组均小于0.1；仅epsilon=1、0.1各三个seed，不是六个epsilon都有输运参考。来源：`results/e3_slab/transport_check.csv`。\n')
table('E3：扩散极限（三seed中位数）',rows('results/e3_slab/summary.csv'),[(k,k) for k in ['epsilon','E_diff','E_q_diff','fick_defect']])
table('E3：组装求积、入流与流平衡（seed11）',rows('results/e3_slab/quadrature_inflow_balance.csv'),[(k,k) for k in ['epsilon','operator_quadrature','rho_change_from_q8','q_change_from_q8','inflow_absolute_L2','q_min','q_max','q_derivative_L2']])
out.append('入流L2为左右各权重1/2的绝对范数；流导数为绝对L2。8→16→32是组装角平均敏感性，区别于测试64→128；主8点数据不替换。来源为 `results/e3_slab/` 下对应CSV。\n')
p5=[]
for p in sorted((ROOT/'results/p5_cutoff_audit').glob('*/*.json')):
    r=json.loads(p.read_text());r['source']=str(p.relative_to(ROOT));p5.append(r)
table('E4：P5相邻截断新计算（seed11，epsilon=1e-3）',p5,[(k,k) for k in ['rcond','relative_l2_f','relative_l2_rho','rank','coefficient_norm','num_rows']])
out.append('两次新计算已完成，但缺同一当前代码下的1e-6基线和独立残差；不能将旧冻结1e-6误差直接与新计算拼接为纯阈值趋势。旧seed11、epsilon=1e-3的Ef=0.0354834、Erho=0.00353813；差异尚待同版本控制。\n')
out.append('权重审计更正：四分量路径实际输出五个内部残差（宏观、even和、odd和、even差、odd差）；当前runner将全部内部行因子设为1，数组后三项仅在legacy三方程分支使用。入流行先单位L2归一化，再乘sqrt(10)，所以入流残差平方权重为10，不按块行数平均。\n')
out.append('残差审计：旧档案residual_half仅来自增广QR尾项，遗漏截断方向残差，不可用于Cemp或阈值稳健性。2026-09-10代码已修正并通过直接矩阵残差回归；旧数值未追改，新旧诊断不可混用。当前计算仍是训练目标残差，不是独立物理残差。\n')
e5=rows('results/e5_mixed/summary.csv')
table('E5：预设全档结果（三seed中位数）',e5,[(k,k) for k in ['method','Nang_positive','seeds','Nrow','Ef','Ef_min','Ef_max','Erho','EF','cost_seconds']])
metrics=rows('results/e5_mixed/physical_metrics.csv')
ref=[]
for key in ['Ef','Erho','EF']:
    vals=[float(r['acceptance_method_error_'+key]) for r in metrics];delta=float(metrics[0]['acceptance_delta_'+key])
    ref.append(dict(metric=key,delta=delta,min_error=min(vals),ratio=delta/min(vals),passed=delta<.1*min(vals)))
table('E5：共同粗网格参考检查（不认证主表细网格）',ref,[(k,k) for k in ['metric','delta','min_error','ratio','passed']])
table('E5：主表细网格f参考检查（含参考插值影响）',metrics,[(k,k) for k in ['method','Nang_positive','seed','main_grid_reference_ratio_Ef','main_grid_reference_pass_Ef']])
table('E5：逐方法—预算—seed分量级验收',metrics,[(k,k) for k in ['method','Nang_positive','seed','reference_ratio_Ef','reference_pass_Ef','reference_ratio_Erho','reference_pass_Erho','reference_ratio_EF','reference_pass_EF']])
out.append('上述验收采用共同粗空间/角网格及插值后的细参考作为统一分母；密度和流先用各自原生角求积计算，再做空间插值。该共同网格验收误差与主表细网格误差分列保存，不能混用。比值阈值不是实际误差下界。\n')
out.append('E5密度核对：主表Erho是重构场f的Gauss角平均误差，而非MM单独宏观变量。原始MM保存的rho也是f的梯形积分，不能拿它与Gauss积分之差当作<g>。旧档案未保存宏观/g场或系数，D0及精确导数Rtr无法直接恢复；明确标为缺失，不设为零。\n')
out.append('E3已完成计算和代表性后处理，无需重复四个新增解。epsilon=1、seed11入流相对误差为sqrt(2)×0.0594806≈8.412%，流采样相对极差约6.745e-4。边界并非精确满足；流近常数不能代替边界或体误差。\n')
out.append('来源：`results/e5_mixed/summary.csv`、`physical_metrics.csv`。rho/F按各参考原生角求积计算后在空间比较；f比较包含角插值。24组全部计算完成不等于参考与求积验收完成。成本是现有批次实测记录，混有全量/分批组装且未隔离复测，不用于加速倍数。\n')
out.append('## 内容要求与交付状态\n\n| 要求 | 填写结论 | 状态 |\n|---|---|---|\n| E1可靠性 | 满秩无截断，阈值及变化已列 | 已填 |\n| E2对应表 | parity、特征扫描、配点扫描分列，旧初始化不改标 | 已填；旧参数哈希未归档 |\n| E3输运/扩散/求积 | 数值及范围见对应表，原8点保留 | 已填；入流平衡限代表seed/epsilon |\n| E4权重 | sqrt(10)作用于归一化入流；五个内部残差单位因子 | 已核对 |\n| E4截断 | 两个新阈值落盘，旧1e-6不可直接混比 | 缺同版本基线与独立残差 |\n| E5缩放 | 全局f=rho+g，不是局部kappa*g | 已核对 |\n| E5损失 | 单位行残差平方和；增加角点会改变相对边界权重 | 已核对，限制归因 |\n| E5参考 | 按最小方法误差逐量验收 | f仍不足；不能整体通过 |\n| E5角求积/计时 | 无已完成加密和隔离三次计时证据 | 待完成 |\n\n结论：E1支持固定空间最小增益，E2支持parity表示贡献，E3支持固定非制造slab的扩散极限趋近；E5不能据当前低预算结果宣称OE普遍节省角信息，P6微观场局限继续保留。\n')
out.insert(1, LATEST_NOTICE)
(ROOT/'docs/e1_e5_filled_tables_2026_09_09.md').write_text('\n'.join(out))
print('Wrote docs/e1_e5_filled_tables_2026_09_09.md')
