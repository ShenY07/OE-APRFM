"""Index current datasets and model files after generating or organizing assets."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def record(path):
    return dict(path=path.relative_to(ROOT).as_posix(),bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def main():
    files=[]
    for folder in ('data/references','models/oe_apnn','models/oe_aprfm'):
        files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and p.suffix in ('.npz','.pt','.json') and p.name not in ('manifest.json','index.json'))
    files.extend((ROOT/'results/comparisons/oe_aprfm/fields').glob('*.npz'))
    records=[record(p) for p in sorted(set(files))]
    (ROOT/'data/manifest.json').write_text(json.dumps(records,indent=2)+'\n')
    rows=[]
    for problem in ('p1','p2','p3','p4','p5'):
        for eps in (1.,.001):
            for seed in (7,11,17):
                rf=f'models/oe_aprfm/baseline/{problem}/eps_{eps:.0e}/seed_{seed}'
                nn=f'models/oe_apnn/baseline/{problem}/eps_{eps:.0e}/steps_20000/seed_{seed}'
                if not (ROOT/rf/'model.npz').exists():
                    raise FileNotFoundError(rf)
                assert (ROOT/nn/'r.pt').exists() and (ROOT/nn/'j.pt').exists()
                rows.append(dict(problem=problem,epsilon=eps,seed=seed,oe_aprfm=rf,oe_apnn=nn,
                                 definition='square_hole_manufactured' if problem=='p5' else problem))
    with (ROOT/'models/comparison_index.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");w.writeheader();w.writerows(rows)
    print(f'Indexed {len(records)} assets and {len(rows)} paired model cases')


if __name__=='__main__':main()
