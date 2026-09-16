# Separate publication panels

Every panel is title-free and exported as PDF and PNG. Panel semantics are encoded in filenames.

## Produced PDF panels

- `fig00_direct_rfm_fixed_discretization_error.pdf`
- `fig00_direct_rfm_macroscopic_residual_response.pdf`
- `fig01_1d_manufactured_error_vs_knudsen.pdf`
- `fig01_2d_manufactured_error_vs_knudsen.pdf`
- `fig01_shared_error_legend.pdf`
- `fig02_1d_manufactured_error_vs_features.pdf`
- `fig02_1d_manufactured_rank_fraction_vs_features.pdf`
- `fig02_2d_manufactured_error_vs_features.pdf`
- `fig02_2d_manufactured_rank_fraction_vs_features.pdf`
- `fig02_shared_epsilon_legend.pdf`
- `fig03_parity_approximation_space.pdf`
- `fig03_parity_feature_budget_kinetic_error_vs_ncoef.pdf`
- `fig03_projection_rescaling_formulation.pdf`
- `fig04_1d_manufactured_accuracy_cost.pdf`
- `fig04_2d_manufactured_accuracy_cost.pdf`
- `fig04_shared_methods_legend.pdf`
- `fig05_heterogeneous_1d_density_comparison.pdf`
- `fig05_heterogeneous_1d_oe_aprfm_phase_space.pdf`
- `fig05_heterogeneous_1d_reference_phase_space.pdf`
- `fig05_heterogeneous_1d_scattering_coefficient.pdf`
- `fig06_perforated_absolute_density_error.pdf`
- `fig06_perforated_density_line_cut_y_0p55.pdf`
- `fig06_perforated_exact_density.pdf`
- `fig06_perforated_oe_aprfm_density.pdf`
- `fig07_p5_absolute_density_error_eps_1e-3.pdf`
- `fig07_p5_deterministic_reference_density_eps_1e-3.pdf`
- `fig07_p5_oe_aprfm_density_eps_1e-3.pdf`
- `fig07_p5_scattering_coefficient.pdf`
- `fig08_1d_manufactured_observed_computational_scaling.pdf`
- `fig08_2d_manufactured_observed_computational_scaling.pdf`
- `fig08_shared_timing_legend.pdf`

## Direct-RFM limitation panels

The two `fig00_direct_rfm_*` panels use P1 on $10^{-6}\le\varepsilon\le1$. The residual-response slope is one; the error curve is the median of genuine seeds 11, 23, and 37 under a fixed 128-coefficient, 1088-row discretization.

## P5 certification note

The four `fig07_p5_*` panels use the boundary-driven smooth P5 problem at $\varepsilon=10^{-3}$. The deterministic density is level C; the OE-APRFM panel uses seed 11. Reference and OE densities share identical color limits. Obsolete source-driven P5 panels are retained under `archive_obsolete_p5_source_driven_2026_08_23/`.
