# Random-feature resolution synchronization

The authoritative feature-resolution results are Table S5 and tables_frozen/feature_resolution_seedwise.csv. Each seedwise row includes its source JSON path. Tables and Fig. 2 use scripts/feature_resolution_data.py; selection is by fixed protocol, not minimum error.

Use epsilon=1e-3, seeds 11/23/37, J=32/64/128 and rcond=1e-12. P1 fixes 2944 residual rows; P3 fixes 13808 rows and uses the final normalized four-component representation. These are exactly the OE-APRFM runs in Table 7. Equal J across dimensions does not mean equal coefficient count, and equal dimensions do not establish equal runtime.

Replace the legacy P3 statement 9.04e-2 -> 5.08e-3 with 3.48e-2 -> 1.16e-3 as J increases from 32 to 128 (128 to 512 coefficients). P1 rank fractions are 1, 1, 0.90234375; P3 fractions are all 1. No claim about other Knudsen regimes or J=256 is made from this sweep. Legacy results/feature_convergence files remain historical records, not the current Fig. 2 source.

The principal P3 configurations in Tables 1/2/8 retain their separately declared 15856-row protocol; they must not be substituted into this fixed-13808-row sweep. No new simulations were run.
