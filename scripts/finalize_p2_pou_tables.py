"""Validate corrected P2 results and rebuild the frozen table bundle."""
import hashlib
import json
import subprocess
import sys
import re
from pathlib import Path
import numpy as np
from p2_corrected_results import corrected_records, ROOT


def main():
    entries=corrected_records()
    comparisons=[]
    for path,r in entries:
        old_path=ROOT/f"results/consistency/p2/p2_oe_aprfm_eps_{r['epsilon']:.0e}_seed_{r['seed']}_fixed.json"
        old=json.loads(old_path.read_text())
        for key in ('epsilon','seed','partitions','features_per_patch','scale','rcond','block_weights','collocation','reference_level','evaluation_grid'):
            assert r[key]==old[key],(key,r[key],old[key])
        with np.load(path.with_suffix('.npz')) as z:
            v=z['velocity'];w=np.full(len(v),2/(len(v)-1));w[[0,-1]]/=2
            ef=np.sqrt(np.sum((z['f']-z['reference_f'])**2*w)/np.sum(z['reference_f']**2*w))
            er=np.linalg.norm(z['rho']-z['reference_rho'])/np.linalg.norm(z['reference_rho'])
            np.testing.assert_allclose([ef,er],[r['relative_l2_f'],r['relative_l2_rho']],rtol=1e-12)
        comparisons.append(dict(epsilon=r['epsilon'],seed=r['seed'],old_E_f=old['relative_l2_f'],new_E_f=r['relative_l2_f'],old_E_rho=old['relative_l2_rho'],new_E_rho=r['relative_l2_rho']))
    directory=ROOT/'results/pou_fix_2026_09_08'
    (directory/'comparison.json').write_text(json.dumps(comparisons,indent=2))
    subprocess.run([sys.executable,str(ROOT/'scripts/build_frozen_manuscript_tables.py')],check=True,cwd=ROOT)
    before=directory/'tables_before';after=ROOT/'results/requirement_2026_08_20/tables_frozen'
    changed=[p.name for p in sorted(after.iterdir()) if p.is_file() and (not (before/p.name).exists() or p.read_bytes()!=(before/p.name).read_bytes())]
    manifest=dict(changed_table_files=changed,results=[str(p.relative_to(ROOT)) for p,_ in entries],source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src/modules/function_space.py',ROOT/'scripts/run_p2_oe_aprfm.py')})
    tex=(after/'all_available_tables.tex').read_text()
    labels=re.findall(r'\\label\{([^}]+)\}',tex)
    assert len(labels)==13 and len(labels)==len(set(labels))
    for match in re.finditer(r'\\begin\{tabular\}\{([^}]+)\}(.*?)\\end\{tabular\}',tex,re.S):
        columns=len(re.findall('[lcr]',match[1]))
        for line in match[2].splitlines():
            if '&' in line and r'\multicolumn' not in line:
                assert line.count('&')+1==columns,line
    summary=[]
    for eps in (1.,.001):
        group=[r for r in comparisons if r['epsilon']==eps]
        summary.append(dict(epsilon=eps,**{k:float(np.median([r[k] for r in group])) for k in ('old_E_f','new_E_f','old_E_rho','new_E_rho')}))
    manifest['median_comparison']=summary
    (directory/'manifest.json').write_text(json.dumps(manifest,indent=2))
    report=['# PoU 修复后的冻结表更新', '',
            '已按原参数重算 P2 两个 ε、三个种子；权重未调整。以下均为三种子中位数。', '',
            '| ε | 原 Ef | 新 Ef | 原 Eρ | 新 Eρ |',
            '|---|---:|---:|---:|---:|']
    for row in summary:
        report.append('| '+ ' | '.join(f'{row[k]:.6g}' for k in ('epsilon','old_E_f','new_E_f','old_E_rho','new_E_rho'))+' |')
    report += ['', '修正项：', '',
               '- 表 1 的 P2 Nrow 从手写的 2944 更正为实际记录的 3008。',
               '- 表 5 的 P2 数值改用新六组结果，合并表和 CSV 同步重建。',
               '- 表格与图 5 改为共享同一新结果来源；密度图使用求积密度，取代简单角平均。',
               '- 74 份相关 P1/P6 表格来源已核对为单分区，本次无需重算。P3/P4 配置行数与记录一致。',
               '', '验证：逐组配置一致；NPZ 可复算 JSON 误差；13 个唯一表标签和各行列数检查通过。',
               '环境没有 pdflatex，未进行 LaTeX 排版编译。旧数据、旧表和旧图均保留。',
               '参考场由历史 NPZ 恢复，梯形求积复现旧密度和误差；未重新认证参考解。',
               '本轮只更新多分区修复影响的 P2，不改变含时 P7 的未达标结论。',
               '', '变更文件：', '']
    report += ['- `'+name+'`' for name in changed]
    (ROOT/'docs/pou_fixed_table_update_2026_09_08.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
