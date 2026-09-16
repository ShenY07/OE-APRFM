# Current standalone experiment panels

Blue: OE/parity; orange: MM/unconstrained. Bands show min/max, not confidence intervals. No panel titles; labels identify axes only.

Only PDFs produced by this script enter the ZIP; old incompatible P7 and archived plots are excluded. E4 is a sensitivity appendix, E5 uses the original reference. E6 error-evolution panels are supplementary, not a replacement for the compact four-row comparison table.

Naming: distribution_error = Ef; density_error = Erho; physical_flux_error = EF; scaled_current_error = Eq. File names do not distinguish physical quantities by letter case alone.

- `E1_fixed_space_minimum_gain.pdf`: 32-dimensional fixed space, no truncation; epsilon=0 drawn as horizontal limit, not on log axis.
- `E2_parity_budget_distribution_error.pdf`: Same J matched rows/coefficients; rows vary across J. Three-seed median and range; old initialization.
- `E2_parity_budget_density_error.pdf`: Same J matched rows/coefficients; rows vary across J. Three-seed median and range; old initialization.
- `E2_p1_fixed_collocation_distribution_error_vs_features.pdf`: Audited fixed-collocation records; P3 four-component implementation.
- `E2_p3_fixed_collocation_distribution_error_vs_features.pdf`: Audited fixed-collocation records; P3 four-component implementation.
- `E3_slab_diffusion_density_error.pdf`: Fixed budget, three-seed median/range. Nonmanufactured slab.
- `E3_slab_diffusion_scaled_current_error.pdf`: Fixed budget, three-seed median/range. Nonmanufactured slab.
- `E3_slab_fick_defect.pdf`: Fixed budget, three-seed median/range. Nonmanufactured slab.
- `E4_P5_seed11_same_version_distribution_relative_L2_error.pdf`: epsilon=1e-3; same-version cutoff sensitivity, not three-seed main results. Residual is complete normalized training residual, not independent physical residual; reference adequacy remains to be checked.
- `E4_P5_seed11_same_version_density_relative_L2_error.pdf`: epsilon=1e-3; same-version cutoff sensitivity, not three-seed main results. Residual is complete normalized training residual, not independent physical residual; reference adequacy remains to be checked.
- `E4_P5_seed11_same_version_rank.pdf`: epsilon=1e-3; same-version cutoff sensitivity, not three-seed main results. Residual is complete normalized training residual, not independent physical residual; reference adequacy remains to be checked.
- `E4_P5_seed11_same_version_coefficient_norm.pdf`: epsilon=1e-3; same-version cutoff sensitivity, not three-seed main results. Residual is complete normalized training residual, not independent physical residual; reference adequacy remains to be checked.
- `E4_P5_seed11_same_version_normalized_training_residual_RMS.pdf`: epsilon=1e-3; same-version cutoff sensitivity, not three-seed main results. Residual is complete normalized training residual, not independent physical residual; reference adequacy remains to be checked.
- `E5_budget_dependence_distribution_error_old_reference_pending_reassessment.pdf`: All eight presets retained. Original reference; new references not yet postprocessed. Actual rows include repeated equations and changing block weights, not independent information. Positive budgets OE/MM each 8,16,32,64 in order.
- `E5_budget_dependence_density_error_old_reference_pending_reassessment.pdf`: All eight presets retained. Original reference; new references not yet postprocessed. Actual rows include repeated equations and changing block weights, not independent information. Positive budgets OE/MM each 8,16,32,64 in order.
- `E5_budget_dependence_physical_flux_error_old_reference_pending_reassessment.pdf`: All eight presets retained. Original reference; new references not yet postprocessed. Actual rows include repeated equations and changing block weights, not independent information. Positive budgets OE/MM each 8,16,32,64 in order.
- `E5_seed11_repeated_wall_time_not_speedup.pdf`: Seed11 only, three repetitions, feature+assembly+solve. Serial queue does not guarantee isolated host. Not paired with three-seed median errors.
- `E6_eps1e+00_ct_distribution_error_three_times.pdf`: Three actually evaluated output times only; seed median/range. CT includes time error; BE compares same time step. CT curves may overlap.
- `E6_eps1e+00_be_distribution_error_three_times.pdf`: Three actually evaluated output times only; seed median/range. CT includes time error; BE compares same time step. CT curves may overlap.
- `E6_eps1e-03_ct_distribution_error_three_times.pdf`: Three actually evaluated output times only; seed median/range. CT includes time error; BE compares same time step. CT curves may overlap.
- `E6_eps1e-03_be_distribution_error_three_times.pdf`: Three actually evaluated output times only; seed median/range. CT includes time error; BE compares same time step. CT curves may overlap.
- `E6_discrete_diffusion_fixed_dt.pdf`: Difference from BE diffusion, not RF error or CT total error.
- `E6_time_refinement_seed11_CT.pdf`: T=0.2 fixed, 50/100/200 BE steps; seed11 only.
- `E6_eps1e+00_seed11_rho_evolution.pdf`: OE seed11 actual saved fields; q(0)=0 retained. Time colors consistent across field panels; not a multi-method comparison.
- `E6_eps1e+00_seed11_q_evolution.pdf`: OE seed11 actual saved fields; q(0)=0 retained. Time colors consistent across field panels; not a multi-method comparison.
- `E6_eps1e+00_seed11_cosine_amplitude_CT_BE.pdf`: Saved output times only; native reference angular moments. OE and BE may overlap; amplitude exposes background-masked time error.
- `E6_eps1e-03_seed11_rho_evolution.pdf`: OE seed11 actual saved fields; q(0)=0 retained. Time colors consistent across field panels; not a multi-method comparison.
- `E6_eps1e-03_seed11_q_evolution.pdf`: OE seed11 actual saved fields; q(0)=0 retained. Time colors consistent across field panels; not a multi-method comparison.
- `E6_eps1e-03_seed11_cosine_amplitude_CT_BE.pdf`: Saved output times only; native reference angular moments. OE and BE may overlap; amplitude exposes background-masked time error.

## Sources
- `results/comparison_repairs_2026_09_10/p5_1e-5/p5_oe_aprfm_eps_1e-03_seed_11.json`
- `results/comparison_repairs_2026_09_10/p5_1e-6/p5_oe_aprfm_eps_1e-03_seed_11.json`
- `results/comparison_repairs_2026_09_10/p5_1e-7/p5_oe_aprfm_eps_1e-03_seed_11.json`
- `results/comparison_repairs_2026_09_10/timing_mm_32_0/mm_s11_a32/metadata.json`
- `results/comparison_repairs_2026_09_10/timing_mm_32_1/mm_s11_a32/metadata.json`
- `results/comparison_repairs_2026_09_10/timing_mm_32_2/mm_s11_a32/metadata.json`
- `results/comparison_repairs_2026_09_10/timing_mm_64_0/mm_s11_a64/metadata.json`
- `results/comparison_repairs_2026_09_10/timing_mm_64_1/mm_s11_a64/metadata.json`
- `results/comparison_repairs_2026_09_10/timing_mm_64_2/mm_s11_a64/metadata.json`
- `results/comparison_repairs_2026_09_10/timing_oe_32_0/oe_s11_a32/metadata.json`
- `results/comparison_repairs_2026_09_10/timing_oe_32_1/oe_s11_a32/metadata.json`
- `results/comparison_repairs_2026_09_10/timing_oe_32_2/oe_s11_a32/metadata.json`
- `results/comparison_repairs_2026_09_10/timing_oe_64_0/oe_s11_a64/metadata.json`
- `results/comparison_repairs_2026_09_10/timing_oe_64_1/oe_s11_a64/metadata.json`
- `results/comparison_repairs_2026_09_10/timing_oe_64_2/oe_s11_a64/metadata.json`
- `results/e3_slab/summary.csv`
- `results/e5_mixed/summary.csv`
- `results/e6_mm_comparison/seedwise.csv`
- `results/e6_periodic/accuracy_summary.csv`
- `results/e6_periodic/factorized_check/s11_eps1e+00_dt2e-03_J128_q64/fields.npz`
- `results/e6_periodic/factorized_check/s11_eps1e-03_dt2e-03_J128_q64/fields.npz`
- `results/e6_periodic/references/eps1e+00_dt2e-03_n256.npz`
- `results/e6_periodic/references/eps1e-03_dt2e-03_n256.npz`
- `results/e6_periodic/time_refinement.csv`
- `results/minimum_gain/quadrature_check.csv`
- `results/requirement_2026_08_20/efficiency_raw/oe_p1_J128/p1_oe_aprfm_eps_1e-03_seed_11_efficiency.json`
- `results/requirement_2026_08_20/efficiency_raw/oe_p1_J32/p1_oe_aprfm_eps_1e-03_seed_11_efficiency.json`
- `results/requirement_2026_08_20/efficiency_raw/oe_p1_J64/p1_oe_aprfm_eps_1e-03_seed_11_efficiency.json`
- `results/requirement_2026_08_20/efficiency_raw/oe_p1_seed23_J128/p1_oe_aprfm_eps_1e-03_seed_23_efficiency.json`
- `results/requirement_2026_08_20/efficiency_raw/oe_p1_seed23_J32/p1_oe_aprfm_eps_1e-03_seed_23_efficiency.json`
- `results/requirement_2026_08_20/efficiency_raw/oe_p1_seed23_J64/p1_oe_aprfm_eps_1e-03_seed_23_efficiency.json`
- `results/requirement_2026_08_20/efficiency_raw/oe_p1_seed37_J128/p1_oe_aprfm_eps_1e-03_seed_37_efficiency.json`
- `results/requirement_2026_08_20/efficiency_raw/oe_p1_seed37_J32/p1_oe_aprfm_eps_1e-03_seed_37_efficiency.json`
- `results/requirement_2026_08_20/efficiency_raw/oe_p1_seed37_J64/p1_oe_aprfm_eps_1e-03_seed_37_efficiency.json`
- `results/requirement_2026_08_20/efficiency_raw_four_component_normalized/oe_p3_seed11_J128/p3_oe_aprfm_eps_1e-03_seed_11_efficiency_four_component.json`
- `results/requirement_2026_08_20/efficiency_raw_four_component_normalized/oe_p3_seed11_J32/p3_oe_aprfm_eps_1e-03_seed_11_efficiency_four_component.json`
- `results/requirement_2026_08_20/efficiency_raw_four_component_normalized/oe_p3_seed11_J64/p3_oe_aprfm_eps_1e-03_seed_11_efficiency_four_component.json`
- `results/requirement_2026_08_20/efficiency_raw_four_component_normalized/oe_p3_seed23_J128/p3_oe_aprfm_eps_1e-03_seed_23_efficiency_four_component.json`
- `results/requirement_2026_08_20/efficiency_raw_four_component_normalized/oe_p3_seed23_J32/p3_oe_aprfm_eps_1e-03_seed_23_efficiency_four_component.json`
- `results/requirement_2026_08_20/efficiency_raw_four_component_normalized/oe_p3_seed23_J64/p3_oe_aprfm_eps_1e-03_seed_23_efficiency_four_component.json`
- `results/requirement_2026_08_20/efficiency_raw_four_component_normalized/oe_p3_seed37_J128/p3_oe_aprfm_eps_1e-03_seed_37_efficiency_four_component.json`
- `results/requirement_2026_08_20/efficiency_raw_four_component_normalized/oe_p3_seed37_J32/p3_oe_aprfm_eps_1e-03_seed_37_efficiency_four_component.json`
- `results/requirement_2026_08_20/efficiency_raw_four_component_normalized/oe_p3_seed37_J64/p3_oe_aprfm_eps_1e-03_seed_37_efficiency_four_component.json`
- `results/requirement_2026_08_20/tables_frozen/table_S4_parity_feature_budget.csv`
