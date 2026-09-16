"""Auditable fixed-collocation feature sweep shared by tables and Fig. 2."""
import csv
import json
from pathlib import Path
from collections import defaultdict
from statistics import median

ROOT = Path(__file__).resolve().parents[1]

def feature_records():
    rows = []
    for problem, pattern in (
        ('p1', 'results/requirement_2026_08_20/efficiency_raw/oe_p1_*/*.json'),
        ('p3', 'results/requirement_2026_08_20/efficiency_raw_four_component_normalized/oe_p3_*/*.json'),
    ):
        for path in sorted(ROOT.glob(pattern)):
            r = json.loads(path.read_text())
            assert r['epsilon'] == 1e-3 and r['rcond'] == 1e-12
            assert r['problem'] == problem
            assert r['num_rows'] == (2944 if problem == 'p1' else 13808)
            assert (r.get('variant') == 'full' if problem == 'p1' else r.get('angular_representation') == 'four_component')
            rows.append(dict(problem=problem, epsilon=r['epsilon'], representation='two_component' if problem == 'p1' else 'four_component',
                features_per_patch=r['features_per_patch'], seed=r['seed'], num_columns=r['num_columns'], num_rows=r['num_rows'],
                rcond=r['rcond'], relative_l2_f=r['relative_l2_f'], relative_l2_rho=r['relative_l2_rho'], rank=r['rank'],
                rank_fraction=r['rank']/r['num_columns'], source=str(path.relative_to(ROOT))))
    groups = defaultdict(list)
    for r in rows:
        groups[r['problem'], r['features_per_patch']].append(r)
    assert set(groups) == {(p,j) for p in ('p1','p3') for j in (32,64,128)}
    for (p,j), members in groups.items():
        assert sorted(r['seed'] for r in members) == [11,23,37]
        assert all(r['num_columns'] == (2 if p == 'p1' else 4)*j for r in members)
    return rows

def feature_summary():
    groups = defaultdict(list)
    for r in feature_records():
        groups[r['problem'], r['features_per_patch']].append(r)
    rows = []
    for key, members in sorted(groups.items()):
        row = {k: members[0][k] for k in ('problem','epsilon','representation','features_per_patch','num_columns','num_rows','rcond')}
        row['seeds'] = len(members)
        for k in ('relative_l2_f','relative_l2_rho','rank_fraction'):
            row[k] = median(r[k] for r in members)
        rows.append(row)
    return rows

def export(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for name, rows in [('table_3_feature_resolution_full.csv', feature_summary()), ('feature_resolution_seedwise.csv', feature_records())]:
        with (directory/name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
