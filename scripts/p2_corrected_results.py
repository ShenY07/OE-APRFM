"""Single validated source for post-PoU-fix P2 manuscript results."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DIRECTORY=ROOT/'results/pou_fix_2026_09_08/p2'


def corrected_records():
    entries=[(p,json.loads(p.read_text())) for p in sorted(DIRECTORY.glob('*_pou_b.json'))]
    expected={(eps,seed) for eps in (1.,.001) for seed in (11,23,37)}
    keys=[(r['epsilon'],r['seed']) for _,r in entries]
    if len(keys)!=6 or set(keys)!=expected:
        raise ValueError('P2 corrected results require exactly two epsilons and seeds 11,23,37; no legacy fallback')
    for p,r in entries:
        assert r['partitions']==[2,4] and r['features_per_patch']==64
        assert r['rcond']==1e-6 and r['evaluation_grid']==[1024,512]
        assert r['num_rows']==3008 and r['num_columns']==1024
        assert p.with_suffix('.npz').exists()
    return entries
