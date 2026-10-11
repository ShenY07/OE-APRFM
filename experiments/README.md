# Numerical experiments

Run from the repository root with `PYTHONPATH=src:.` and the environment
settings in the root README. Full budgets and reference prerequisites are in
[the reproduction guide](../docs/reproduction.md).

| Entry point | Purpose |
| --- | --- |
| `run_p1_oe_aprfm.py` | P1 and P6 manufactured 1D problems |
| `run_p2_oe_aprfm.py` | P2 heterogeneous 1D transport |
| `run_p3_oe_aprfm.py` | P3–P5 two-dimensional problems |
| `diagnose_oe_gain.py` | E1 fixed-space minimum gain |
| `run_e3_slab.py`, `run_e5_mixed.py` | Steady slab and mixed-scale studies |
| `run_e6_periodic.py`, `reference_e6.py` | Transient experiment and Fourier references |
| `run_mm_aprfm_1d.py`, `run_mm_aprfm_2d.py`, `run_e6_mm.py` | Matched micro–macro comparisons |
| `run_rfm_1d.py`, `run_oe_si_dsa.py` | Direct RFM and deterministic transport |
| `run_angle_comparison.py`, `run_angle_campaign.py` | Angular-domain comparison and campaign |
| `generate_reference.py`, `exact_references.py` | Numerical and manufactured references |
| `retrain_oe_apnn.py` | Neural baseline training campaign |

`e6_protocol.py` holds the shared transient protocol. P labels identify
problems; E labels identify manuscript studies. Preserve these labels in run
records. New outputs belong under `results/`, with separate directories for
each problem, configuration, and seed. Run the matching `scripts/report_*.py`
entry point to aggregate completed runs.
