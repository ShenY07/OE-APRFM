"""One-time migration of known E6 runs; no trajectory is recomputed."""
import json,csv,hashlib
from pathlib import Path
import numpy as np
from e6_protocol import PROTOCOL,IMPLEMENTATION,digest,validate
from run_e6_periodic import Space
root=Path(__file__).resolve().parents[1]/'results/e6_periodic'
rows=[]
for p in sorted(root.glob('*/*/result.json')):
    r=json.loads(p.read_text());valid=p.parent.parent.name in ('factorized_check','time_check','feature_check','quadrature_check')
    with np.load(p.parent/'fields.npz') as d:
        params=d['parameters'];np.testing.assert_array_equal(params,Space(r['J'],r['seed']).p)
    r.update(protocol_id=PROTOCOL,implementation_id=IMPLEMENTATION if valid else 'invalid_precomposed_update',
             config_hash=digest(r),feature_hash=hashlib.sha256(params.tobytes()).hexdigest(),valid_for_e6=valid,
             provenance_note='Retrospective metadata from known execution batches and saved parameters; not a historical source-code hash')
    r['feature_nested_check']=bool(np.array_equal(Space(256,r['seed']).p[:128],Space(128,r['seed']).p))
    p.write_text(json.dumps(r,indent=2)+'\n')
    if valid:
        assert validate(p),str(p)
        rows.append(dict(run=str(p.parent.relative_to(root)),rank=r['rank'],valid_for_e6=True,
            feature_nested=r['feature_nested_check'],
            max_mass_drift=max(h['mass_drift'] for h in r['history']),
            max_periodic_r=max(h['periodic_r'] for h in r['history']),
            max_periodic_j=max(h['periodic_j'] for h in r['history']),
            constant_r_error=r['constant_step_regression']['constant_r_error'],
            constant_j_error=r['constant_step_regression']['constant_j_error'],
            reference_status='pending',test_refinement_status='pending'))
assert len(rows)==20
with (root/'trajectory_audit.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('Validated 20 trajectories; excluded 2 invalid runs; nested parameters checked.')
