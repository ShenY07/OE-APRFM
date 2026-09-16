"""Rebuild saved RF solutions to add parity errors without replacing old metrics."""
import json
from pathlib import Path
from functools import partial
from run_p7_transient_pilot import build,reference,errors
from p7_consistent_features import ConsistentFeatures
import numpy as np

p=Path('results/p7_transient_pilot/consistent_pou_sweep.json')
rows=json.loads(p.read_text())
backup=p.with_name('consistent_pou_sweep_before_parity_errors.json')
if not backup.exists():backup.write_text(p.read_text())
for r in rows:
 if all('E_j' in s and 'E_r' in s for s in r['snapshots']):continue
 f,_=build(r['epsilon'],r['J'],r['seed'],'oe_ap',samples=r['samples'],nv=r['nv'],
     repeat_macro=r['repeat_macro'],block_average=r['block_average'],
     boundary_samples=r['boundary_samples'],initial_samples=r['initial_samples'],
     feature_factory=partial(ConsistentFeatures,partitions=tuple(r['partitions']),scale=r['scale']))
 fresh=errors(f,reference(r['epsilon'],256,16,800),r['epsilon'])
 for old,new in zip(r['snapshots'],fresh):
  np.testing.assert_allclose([new['E_f'],new['E_rho']],[old['E_f'],old['E_rho']],rtol=2e-5,atol=1e-9)
  old.update(E_r=new['E_r'],E_j=new['E_j'])
 p.write_text(json.dumps(rows,indent=2))
 print(r['partitions'],r['epsilon'],fresh[-1],flush=True)
