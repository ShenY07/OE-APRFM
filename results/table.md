\begin{table}[!htb]
\centering
\small
\setlength{\tabcolsep}{5pt}
\caption{Principal OE-APRFM discretization sizes.
The two-dimensional fixed-budget and feature/cost studies use the same physical-inflow trace assembly and angular-midpoint evaluation.
Feature-refinement configurations are detailed in Table \ref{tab:feature-resolution}.}
\label{tab:configurations}
\begin{tabular}{lrr}
\toprule
Problem & $N_{\rm coef}$ & $N_{\rm row}$\\
\midrule
1D manufactured & 128 & 2944\\
2D manufactured ($J=128$) & 512 & 13808\\
Isotropic-inflow slab & 128 & 2944\\
Periodic transport & 256 & 4288\\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[!htb]
\centering
\small
\setlength{\tabcolsep}{5pt}
\caption{Residual-formulation and trial-space comparison for the one-dimensional manufactured problem (128 coefficients and seed $11$).
Residual counts $N_{\rm row}$ differ between configurations.
The unprojected control is defined on the positive half-angle domain and reconstructed as
$f_h(x,v)=\widetilde r_h(x,|v|)+\varepsilon\operatorname{sgn}(v)\widetilde j_h(x,|v|)$.}
\label{tab:mechanism}
\begin{tabular}{lcccc}
\toprule
Configuration
& $N_{\rm row}$
& $E_f(1)$
& $E_f(10^{-3})$
& $E_f(10^{-6})$ \\
\midrule
Direct RFM
& 1088
& $\boldsymbol{9.80\times10^{-11}}$
& $1.28\times10^{-6}$
& $1.15\times10^{-3}$ \\
Unscaled OE
& 1984
& $5.73\times10^{-7}$
& $1.54\times10^{-3}$
& $2.91\times10^{-2}$ \\
Unprojected OE
& 2944
& $1.74\times10^{-6}$
& $1.01\times10^{-6}$
& $1.01\times10^{-6}$ \\
\addlinespace[3pt]
\textbf{OE-APRFM}
& 2944
& $5.60\times10^{-7}$
& $\boldsymbol{5.21\times10^{-7}}$
& $\boldsymbol{5.24\times10^{-7}}$ \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[!htb]
\centering\small
\setlength{\tabcolsep}{7pt}
\caption{Fixed-budget errors for the one- and two-dimensional manufactured solutions. Values are three-seed medians. The $\varepsilon$-dependent manufactured sources test cross-scale approximation, not a fixed-data AP limit.
The two-dimensional results use 512 coefficients and 13,808 residual rows, with the same physical-inflow trace assembly and angular-midpoint evaluation as the feature/cost studies.}
\label{tab:knudsen-accuracy}
\begin{tabular}{lccc}
\toprule
Problem & $\varepsilon$ & $E_f$ & $E_\rho$\\
\midrule
1D manufactured & $1$ & $2.71\times10^{-8}$ & $2.64\times10^{-9}$\\
 & $10^{-3}$ & $1.205\times10^{-8}$ & $4.73\times10^{-9}$\\
 & $10^{-6}$ & $1.21\times10^{-8}$ & $4.78\times10^{-9}$\\
\addlinespace
2D manufactured & $1$ & $1.274\times10^{-3}$ & $4.26\times10^{-4}$\\
 & $10^{-3}$ & $1.163\times10^{-3}$ & $4.86\times10^{-4}$\\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[!htb]
\centering
\small
\setlength{\tabcolsep}{3.5pt}
\caption{Accuracy and computation time for the manufactured problems at $\varepsilon=10^{-3}$; RFM and APNN results are three-seed medians, and deterministic results are single solves.
$N_{\rm model}$ denotes the number of random-feature coefficients, trainable network parameters, or grid unknowns. OE-APNN uses the new hard-parity 5000-step, 5120-phase-sample CUDA protocol (seeds 7, 11, 17); concurrent GPU times are not directly comparable to timings of other methods. The 2D OE-APRFM entries are from the unified-protocol reruns; their computation times are recorded serial fresh-process measurements on a shared host.}
\label{tab:efficiency-summary}
\begin{tabular}{llcccc}
\toprule
Case & Method & $N_{\rm model}$ & $E_f$ & $E_\rho$ & $T_{\rm comp}$ (s) \\
\midrule
1D & OE-$S_N$-Krylov & 512
   & $1.95\times10^{-11}$ & $1.95\times10^{-11}$ & \textbf{0.08} \\
   & MM-APRFM & 256
   & $6.93\times10^{-11}$ & $6.93\times10^{-11}$ & 3.00 \\
   & OE-APNN & 25474
   & $8.504\times10^{-4}$ & $6.872\times10^{-4}$ & 146.70\\
   & \textbf{OE-APRFM} & 256
   & $\boldsymbol{3.23\times10^{-12}}$
   & $\boldsymbol{2.66\times10^{-12}}$ & 17.86 \\
\addlinespace
2D & OE-$S_N$-Krylov & 9216
   & $8.04\times10^{-4}$ & $8.04\times10^{-4}$ & \textbf{0.77} \\
   & MM-APRFM & 512
   & $\boldsymbol{9.11\times10^{-8}}$
   & $\boldsymbol{9.11\times10^{-8}}$ & 9.63 \\
   & OE-APNN & 25730
   & $2.635\times10^{-3}$ & $1.457\times10^{-3}$ & 184.10\\
   & \textbf{OE-APRFM} & 512
   & $1.163\times10^{-3}$ & $4.86\times10^{-4}$ & 19.61 \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[!htb]
\centering
\small
\setlength{\tabcolsep}{3pt}
\caption{Periodic transport at $t=0.2$ with $\Delta t=0.002$. Values are three-seed medians,
and both RFM methods use 256 coefficients. CT and BE denote continuous-time and same-step backward-Euler transport references.}
\label{tab:transient-comparison}
\resizebox{\linewidth}{!}{%
\begin{tabular}{cccccc}
\toprule
$\varepsilon$ & Method
& $E_f^{\rm CT}$ & $E_f^{\rm BE}$
& $E_\rho^{\rm BE}$ & $E_q^{\rm BE}$\\
\midrule
$1$ & MM-APRFM
& $\boldsymbol{4.430\times10^{-4}}$
& $9.246\times10^{-8}$
& $4.296\times10^{-8}$
& $1.684\times10^{-7}$\\
& \textbf{OE-APRFM}
& $\boldsymbol{4.430\times10^{-4}}$
& $\boldsymbol{7.258\times10^{-9}}$
& $\boldsymbol{3.579\times10^{-9}}$
& $\boldsymbol{3.365\times10^{-8}}$\\
\addlinespace
$10^{-3}$ & MM-APRFM
& $\boldsymbol{3.523\times10^{-4}}$
& $3.951\times10^{-9}$
& $3.950\times10^{-9}$
& $1.354\times10^{-6}$\\
& \textbf{OE-APRFM}
& $\boldsymbol{3.523\times10^{-4}}$
& $\boldsymbol{3.180\times10^{-10}}$
& $\boldsymbol{3.180\times10^{-10}}$
& $\boldsymbol{4.537\times10^{-8}}$\\
\bottomrule
\end{tabular}%
}
\end{table}

\begin{table}[!htb]
\centering\small
\setlength{\tabcolsep}{3.5pt}
\caption{Steady OE-APRFM configurations used in the supplementary studies. $J$ is the number of sampled features per local parity component; $N_{\rm ang}$ denotes the configured angular quadrature order, not the number of angular collocation points. The two-dimensional variable-scattering cutoff study uses physical-direction inflow traces and 58096 residual rows, as detailed in Table \ref{tab:p5-cutoff}.}
\label{tab:supp-configurations}
\begin{tabular}{lccccc}
\toprule
Problem & Partition & $J$ & $N_{\rm coef}$ & $N_{\rm ang}$ & \texttt{rcond}\\
\midrule
1D manufactured & $1\times1$ & 64 & 128 & 8 & $10^{-12}$\\
Isotropic-inflow slab & $1\times1$ & 64 & 128 & 8 & $10^{-12}$\\
1D variable scattering & $2\times4$ & 64 & 1024 & 8 & $10^{-6}$\\
2D manufactured & $1\times1\times1$ & 128 & 512 & 32 & $10^{-12}$\\
Perforated square & $1\times1\times1$ & 128 & 512 & 32 & $10^{-12}$\\
2D variable scattering & $1\times1\times2$ & 128 & 1024 & 16 & $10^{-6}$\\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[!htb]
\centering\small
\setlength{\tabcolsep}{12pt}
\caption{Fixed-space minimum gain for the seed-11, 32-dimensional parity-conforming space. All directions are retained and no SVD truncation is applied. The diagnostic uses its own $H^1\times H^1$ and unweighted inflow norms and refined quadrature; it does not verify Assumption~1 for the main experiment configurations.}
\label{tab:minimum-gain}
\begin{tabular}{ccc}
\toprule
$\varepsilon$ & $\beta_{\varepsilon,m}^{(q)}$ & Rank / dimension\\
\midrule
$1$ & 0.1684021 & 32 / 32\\
$10^{-1}$ & 0.04933323 & 32 / 32\\
$10^{-2}$ & 0.03802870 & 32 / 32\\
$10^{-3}$ & 0.03739774 & 32 / 32\\
$10^{-6}$ & 0.03733276 & 32 / 32\\
$0$ & 0.03733270 & 32 / 32\\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[!htb]
\centering\small
\setlength{\tabcolsep}{3.5pt}
\caption{Random-feature refinement at fixed residual dimension and $\varepsilon=10^{-3}$. Values are medians over seeds 11, 23 and 37 from the same runs as the accuracy-cost tables.}
\label{tab:feature-resolution}
\resizebox{\linewidth}{!}{%
\begin{tabular}{ccccccc}
\toprule
Case & $J$ & $N_{\rm coef}$ & $N_{\rm row}$ & $E_f$ & $E_\rho$ & Rank fraction\\
\midrule
1D & 32 & 64 & 2944 & $5.790\times10^{-5}$ & $2.77\times10^{-5}$ & 1.000\\
 & 64 & 128 & 2944 & $1.205\times10^{-8}$ & $4.73\times10^{-9}$ & 1.000\\
 & 128 & 256 & 2944 & $\boldsymbol{3.227\times10^{-12}}$ & $\boldsymbol{2.66\times10^{-12}}$ & 0.902344\\
\addlinespace
2D & 32 & 128 & 13808 & $3.475\times10^{-2}$ & $1.94\times10^{-2}$ & 1.000 \\
 & 64 & 256 & 13808 & $1.034\times10^{-2}$ & $5.81\times10^{-3}$ & 1.000 \\
 & 128 & 512 & 13808 & $\boldsymbol{1.163\times10^{-3}}$ & $\boldsymbol{4.86\times10^{-4}}$ & 1.000 \\
\bottomrule
\end{tabular}%
}
\end{table}

\begin{table}[!htb]
\centering\small
\setlength{\tabcolsep}{8pt}
\caption{Collocation refinement at $\varepsilon=10^{-3}$ and 256 coefficients. The 2D rows use the half-circle two-component representation.}
\label{tab:collocation}
\begin{tabular}{cccc}
\toprule
Case & $\eta_{\rm col}$ & $E_f$ & $E_\rho$\\
\midrule
1D & 2.078 & $1.35\times10^{-9}$ & $1.35\times10^{-9}$\\
 & 3.934 & $7.73\times10^{-11}$ & $7.71\times10^{-11}$\\
 & 8.121 & $6.02\times10^{-12}$ & $5.68\times10^{-12}$\\
 & 12.246 & $3.03\times10^{-12}$ & $2.43\times10^{-12}$\\
\addlinespace
2D & 2 & $9.57\times10^{-2}$ & $3.58\times10^{-2}$\\
 & 4 & $3.11\times10^{-2}$ & $3.01\times10^{-2}$\\
 & 8 & $1.98\times10^{-2}$ & $1.81\times10^{-2}$\\
 & 12 & $1.89\times10^{-2}$ & $1.71\times10^{-2}$\\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[!htb]
\centering\small
\setlength{\tabcolsep}{3pt}
\caption{Matched-budget basis-construction comparison at $\varepsilon=10^{-3}$. Entries are three-seed medians and $\eta_{\rm col}=N_{\rm row}/N_{\rm coef}$.
The unprojected half-range control uses
$f_h(x,v)=\widetilde r_h(x,|v|)+\varepsilon\operatorname{sgn}(v)\widetilde j_h(x,|v|)$ for evaluation, consistently with its half-range training boundary conditions.
Both reconstructions have even $r_h$ and odd $j_h$. Coefficient and residual counts are matched within each $J$; residual counts vary across budgets.}
\label{tab:parity-budget}
\begin{tabular}{lcccccc}
\toprule
Space & $J$ & $N_{\rm coef}$ & $N_{\rm row}$ & $\eta_{\rm col}$ & $E_f$ & $E_\rho$\\
\midrule
Unprojected & 16 & 32 & 812 & 25.375 & $7.956\times10^{-3}$ & $6.01\times10^{-3}$\\
 & 32 & 64 & 1452 & 22.688 & $1.063\times10^{-4}$ & $5.98\times10^{-5}$\\
 & 64 & 128 & 2944 & 23.000 & $2.823\times10^{-7}$ & $6.29\times10^{-8}$\\
 & 128 & 256 & 6266 & 24.477 & $1.052\times10^{-10}$ & $3.98\times10^{-11}$\\
\addlinespace
Projected & 16 & 32 & 812 & 25.375 & $1.123\times10^{-2}$ & $1.04\times10^{-2}$\\
 & 32 & 64 & 1452 & 22.688 & $6.809\times10^{-5}$ & $4.83\times10^{-5}$\\
 & 64 & 128 & 2944 & 23.000 & $1.205\times10^{-8}$ & $4.73\times10^{-9}$\\
 & 128 & 256 & 6266 & 24.477 & $2.105\times10^{-12}$ & $1.37\times10^{-12}$\\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[!htb]
\centering\small
\setlength{\tabcolsep}{6pt}
\caption{OE-APRFM errors for the manufactured solution with angular dependence.}
\label{tab:angular-manufactured}
\begin{tabular}{ccccc}
\toprule
$\varepsilon$ & $E_f$ & $E_\rho$ & $E_r$ & $E_j$\\
\midrule
$1$ & $2.17\times10^{-5}$ & $5.90\times10^{-6}$ & $1.86\times10^{-5}$ & $1.13\times10^{-4}$\\
$10^{-3}$ & $1.64\times10^{-5}$ & $1.59\times10^{-5}$ & $1.64\times10^{-5}$ & $1.12\times10^{-3}$\\
$10^{-6}$ & $1.64\times10^{-5}$ & $1.59\times10^{-5}$ & $1.64\times10^{-5}$ & $1.12\times10^{-3}$\\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[!htb]
\centering\small
\setlength{\tabcolsep}{6pt}
\caption{Matched-coefficient comparison for the angular-dependent manufactured solution. Entries are three-seed medians.}
\label{tab:angular-mm}
\begin{tabular}{ccccc}
\toprule
$\varepsilon$ & $E_f$, OE & $E_f$, MM & $E_q$, OE & $E_q$, MM\\
\midrule
$1$ & $2.17\times10^{-5}$ & $1.58\times10^{-5}$ & $6.28\times10^{-5}$ & $6.82\times10^{-6}$\\
$10^{-3}$ & $1.64\times10^{-5}$ & $4.82\times10^{-7}$ & $1.62\times10^{-4}$ & $3.34\times10^{-6}$\\
$10^{-6}$ & $1.64\times10^{-5}$ & $4.85\times10^{-7}$ & $1.61\times10^{-4}$ & $3.47\times10^{-6}$\\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[!htb]
\centering\small
\setlength{\tabcolsep}{3pt}
\caption{Two- and four-component angular representations at 512 coefficients. All entries are medians over seeds 11, 23 and 37 from fresh-process reruns of the current implementation, using nominal $16\times16\times16$ collocation parameters and a common angular-midpoint evaluation grid. The four-component runs use physical-inflow trace assembly. Times include feature construction, assembly and solution, but exclude evaluation; runs were serialized on a shared host.}
\label{tab:p34-components}
\begin{tabular}{ccccccc}
\toprule
$\varepsilon$ & Components & $N_{\rm row}$ & $E_f$ & $E_\rho$ & $\kappa_{\rm eff}$ & Time (s)\\
\midrule
\multicolumn{7}{l}{2D manufactured solution}\\
$1$ & 2 & 16160 & $1.418\times10^{-2}$ & $2.13\times10^{-3}$ & $6.92\times10^{7}$ & 46.05\\
 & 4 & 13808 & $1.274\times10^{-3}$ & $4.26\times10^{-4}$ & $1.25\times10^{6}$ & 18.81\\
$10^{-3}$ & 2 & 16160 & $1.217\times10^{-3}$ & $1.10\times10^{-3}$ & $3.53\times10^{8}$ & 52.77\\
 & 4 & 13808 & $1.163\times10^{-3}$ & $4.86\times10^{-4}$ & $7.92\times10^{6}$ & 19.61\\
\addlinespace
\multicolumn{7}{l}{Perforated square}\\
$1$ & 2 & 15840 & $5.358\times10^{-2}$ & $2.04\times10^{-2}$ & $9.39\times10^{7}$ & 39.09\\
 & 4 & 13712 & $5.727\times10^{-2}$ & $1.56\times10^{-2}$ & $1.49\times10^{6}$ & 20.26\\
$10^{-3}$ & 2 & 15840 & $1.767\times10^{-2}$ & $1.37\times10^{-2}$ & $4.25\times10^{8}$ & 45.22\\
 & 4 & 13712 & $2.190\times10^{-2}$ & $1.82\times10^{-2}$ & $7.12\times10^{6}$ & 18.94\\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[!htb]
\centering\small
\setlength{\tabcolsep}{2.8pt}
\caption{Complete accuracy and cost sweep for the one-dimensional manufactured solution at $\varepsilon=10^{-3}$. Random-feature and APNN values are three-seed medians; deterministic values are single solves. Hard-parity OE-APNN uses seeds 7, 11, 17, 4096 interior and 1024 boundary phase samples per step ($N_{\rm phase}=5120$, replacing the previous 640-sample budget); $N_{\rm res}$ in its rows denotes this phase-sample budget, not scalar equation count. Times are concurrent CUDA training wall times on A800 GPUs with three seeds per GPU; other-method times are historical and do not establish a same-hardware speedup.}
\label{tab:supp-efficiency}
\begin{tabular}{llccccc}
\toprule
Method & Budget & $N_{\rm model}$ & $N_{\rm res}$ & $E_f$ & $E_\rho$ & $T_{\rm comp}$ (s)\\
\midrule
OE-$S_N$-Krylov & $32\times16$ & 512 & 512 & $1.95\times10^{-11}$ & $1.95\times10^{-11}$ & 0.08\\
 & $64\times32$ & 2048 & 2048 & $1.51\times10^{-10}$ & $1.51\times10^{-10}$ & 0.19\\
 & $128\times64$ & 8192 & 8192 & $9.80\times10^{-12}$ & $9.80\times10^{-12}$ & 0.94\\
 & $256\times128$ & 32768 & 32768 & $1.44\times10^{-11}$ & $1.44\times10^{-11}$ & 17.65\\
\addlinespace
MM-APRFM & $J=32$ & 64 & 3008 & $5.34\times10^{-11}$ & $5.34\times10^{-11}$ & 2.21\\
 & $J=64$ & 128 & 3008 & $7.25\times10^{-11}$ & $7.25\times10^{-11}$ & 2.57\\
 & $J=128$ & 256 & 3008 & $6.93\times10^{-11}$ & $6.93\times10^{-11}$ & 3.00\\
\addlinespace
OE-APNN & 500 steps & 25474 & 5120 & $3.233\times10^{-3}$ & $2.790\times10^{-3}$ & 20.04\\
  & 2000 steps & 25474 & 5120 & $1.764\times10^{-3}$ & $1.199\times10^{-3}$ & 71.27\\
  & 5000 steps & 25474 & 5120 & $8.504\times10^{-4}$ & $6.872\times10^{-4}$ & 146.70\\
\addlinespace
OE-APRFM & $J=32$ & 64 & 2944 & $5.790\times10^{-5}$ & $2.77\times10^{-5}$ & 17.58\\
 & $J=64$ & 128 & 2944 & $1.205\times10^{-8}$ & $4.73\times10^{-9}$ & 17.69\\
 & $J=128$ & 256 & 2944 & $3.227\times10^{-12}$ & $2.66\times10^{-12}$ & 17.86\\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[!htb]
\centering\small
\setlength{\tabcolsep}{2.8pt}
\caption{Complete accuracy and cost sweep for the two-dimensional manufactured solution at $\varepsilon=10^{-3}$. OE-APRFM uses the four-component representation and 13808 residual rows. Hard-parity OE-APNN uses seeds 7, 11, 17, 4096 interior and 1024 boundary phase samples per step ($N_{\rm phase}=5120$, replacing the previous 640-sample budget); $N_{\rm res}$ in its rows denotes this phase-sample budget, not scalar equation count. Times are concurrent CUDA training wall times on A800 GPUs with three seeds per GPU; deterministic and MM times are historical and do not establish a same-hardware speedup. OE-APRFM uses physical-inflow trace assembly and angular-midpoint evaluation; its errors and recorded CPU computation times are updated together from serial fresh-process reruns on a shared host.}
\label{tab:supp-efficiency-2d}
\begin{tabular}{llccccc}
\toprule
Method & Budget & $N_{\rm model}$ & $N_{\rm res}$ & $E_f$ & $E_\rho$ & $T_{\rm comp}$ (s)\\
\midrule
OE-$S_N$-Krylov & $12^2\times16$ & 2304 & 2304 & $2.98\times10^{-3}$ & $2.98\times10^{-3}$ & 0.33\\
 & $16^2\times16$ & 4096 & 4096 & $1.75\times10^{-3}$ & $1.75\times10^{-3}$ & 0.41\\
 & $24^2\times16$ & 9216 & 9216 & $8.04\times10^{-4}$ & $8.04\times10^{-4}$ & 0.77\\
 & $32^2\times16$ & 16384 & 16384 & $4.57\times10^{-4}$ & $4.57\times10^{-4}$ & 3.24\\
\addlinespace
MM-APRFM & $J=64$ & 128 & 10280 & $4.96\times10^{-6}$ & $4.96\times10^{-6}$ & 4.46\\
 & $J=128$ & 256 & 10280 & $3.65\times10^{-8}$ & $3.65\times10^{-8}$ & 5.61\\
 & $J=256$ & 512 & 10280 & $9.11\times10^{-8}$ & $9.11\times10^{-8}$ & 9.63\\
\addlinespace
OE-APNN & 500 steps & 25730 & 5120 & $2.504\times10^{-2}$ & $1.885\times10^{-2}$ & 23.16\\
  & 2000 steps & 25730 & 5120 & $6.273\times10^{-3}$ & $3.837\times10^{-3}$ & 87.72\\
  & 5000 steps & 25730 & 5120 & $2.635\times10^{-3}$ & $1.457\times10^{-3}$ & 184.10\\
\addlinespace
OE-APRFM & $J=32$ & 128 & 13808 & $3.475\times10^{-2}$ & $1.94\times10^{-2}$ & 14.73\\
 & $J=64$ & 256 & 13808 & $1.034\times10^{-2}$ & $5.81\times10^{-3}$ & 16.26\\
 & $J=128$ & 512 & 13808 & $1.163\times10^{-3}$ & $4.86\times10^{-4}$ & 19.61\\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[!htb]
\centering\small
\setlength{\tabcolsep}{7pt}
\caption{Steady slab transport with isotropic inflow: differences from the diffusion density and scaled flux, together with the interior Fick defect.}
\label{tab:slab-diffusion}
\begin{tabular}{cccc}
\toprule
$\varepsilon$ & $E_{\rho,\rm diff}$ & $E_{q,\rm diff}$ & $D_{\rm Fick}$\\
\midrule
$1$ & $2.538\times10^{-1}$ & $5.765\times10^{-1}$ & $7.201\times10^{-2}$\\
$10^{-1}$ & $5.685\times10^{-2}$ & $1.182\times10^{-1}$ & $3.944\times10^{-3}$\\
$10^{-2}$ & $6.368\times10^{-3}$ & $1.291\times10^{-2}$ & $8.433\times10^{-4}$\\
$10^{-3}$ & $6.424\times10^{-4}$ & $1.302\times10^{-3}$ & $8.634\times10^{-5}$\\
$10^{-4}$ & $6.428\times10^{-5}$ & $1.303\times10^{-4}$ & $8.623\times10^{-6}$\\
$10^{-6}$ & $6.287\times10^{-7}$ & $1.271\times10^{-6}$ & $1.012\times10^{-7}$\\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[!htb]
\centering
\small
\setlength{\tabcolsep}{5pt}
\caption{Mixed-scale transport errors relative to the archived reference. Reference-accuracy diagnostics on the main evaluation grid did not meet the prescribed acceptance criterion for all configurations. These are differences from the archived reference, used to examine budget dependence and excluded from the main accuracy claims; they do not support fine method rankings.
Angular budgets count positive-velocity nodes; $N_{\rm row}$ includes repeated constraints.}
\label{tab:mixed-budget}
\begin{tabular}{lccccc}
\toprule
Method & Angular budget & $N_{\rm row}$ & $E_f$ & $E_\rho$ & $E_F$ \\
\midrule
MM & 8  & 6304  & $1.585\times10^{-1}$ & $9.361\times10^{-2}$ & $7.776\times10^{-1}$ \\
   & 16 & 12352 & $1.279\times10^{-3}$ & $7.904\times10^{-4}$ & $1.707\times10^{-3}$ \\
   & 32 & 24448 & $1.138\times10^{-3}$ & $7.974\times10^{-4}$ & $4.282\times10^{-4}$ \\
   & 64 & 48640 & $1.323\times10^{-3}$ & $1.148\times10^{-3}$ & $1.419\times10^{-3}$ \\
\addlinespace
OE & 8  & 3280  & $6.935\times10^{-1}$ & $7.682\times10^{-2}$ & $1.495$ \\
   & 16 & 6304  & $7.808\times10^{-2}$ & $4.550\times10^{-2}$ & $2.939\times10^{-1}$ \\
   & 32 & 12352 & $3.267\times10^{-3}$ & $3.091\times10^{-3}$ & $1.288\times10^{-3}$ \\
   & 64 & 24448 & $3.224\times10^{-3}$ & $3.121\times10^{-3}$ & $1.131\times10^{-3}$ \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[!htb]
\centering
\small
\setlength{\tabcolsep}{9pt}
\caption{Median, minimum, and maximum computation times (s) over three independent process runs for the one-dimensional mixed-scale problem with seed 11. These timings do not establish an equal-accuracy speedup because the corresponding reference-relative differences are not uniformly accuracy-certified.}
\label{tab:mixed-timing}
\begin{tabular}{lcccc}
\toprule
Method & Positive-angle budget & Median & Minimum & Maximum \\
\midrule
MM & 32 & 202.22 & 202.15 & 204.99 \\
MM & 64 & 356.93 & 344.11 & 357.95 \\
OE & 32 & 114.94 & 114.05 & 116.10 \\
OE & 64 & 133.60 & 129.37 & 145.71 \\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[!htb]
\centering\small
\setlength{\tabcolsep}{7pt}
\caption{OE-APRFM errors for one-dimensional variable scattering and the perforated square. The one-dimensional entries use the six post-PoU-correction runs with consistent assembly and reconstruction, evaluated against the same retained level-B reference; the reference was not regenerated or newly certified.}
\label{tab:extended}
\begin{tabular}{lccc}
\toprule
Problem & $\varepsilon$ & $E_f$ & $E_\rho$\\
\midrule
1D variable scattering & $1$ & $2.41\times10^{-2}$ & $3.34\times10^{-3}$\\
 & $10^{-3}$ & $2.08\times10^{-3}$ & $1.10\times10^{-3}$\\
\addlinespace
Perforated square & $1$ & $5.727\times10^{-2}$ & $1.56\times10^{-2}$\\
 & $10^{-3}$ & $2.190\times10^{-2}$ & $1.82\times10^{-2}$\\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[!htb]
\centering\small
\setlength{\tabcolsep}{4pt}
\caption{Relative-cutoff sensitivity for two-dimensional variable scattering with seed 11 and $\varepsilon=10^{-3}$. The archived source, verified against the recorded hashes, imposes one physical inflow trace per sampled inward direction, consistently with reconstruction. The 58096 rows comprise 10800 macro, 21600 even, 21600 odd and 4096 boundary equations; the historical metadata label \texttt{independent\_pair\_traces\_v2} is misleading. The residual column is the complete normalized least-squares residual RMS. The coefficient norm is $\|\theta\|_2$ after undoing column equilibration $z=D\theta$. The cutoff acts on the equilibrated system; rank reduction does not invalidate the exact least-squares theorem, whose direct coverage does not extend to these truncated solves.}
\label{tab:p5-cutoff}
\begin{tabular}{cccccc}
\toprule
\texttt{rcond} & $E_f$ & $E_\rho$ & $r_{\rm eff}$ & $\|\theta\|_2$ & LS residual RMS\\
\midrule
$10^{-5}$ & $3.195\times10^{-3}$ & $2.699\times10^{-3}$ & 206 & $1.734\times10^3$ & $3.987\times10^{-3}$\\
$10^{-6}$ & $5.416\times10^{-4}$ & $4.117\times10^{-4}$ & 317 & $5.827\times10^3$ & $9.132\times10^{-4}$\\
$10^{-7}$ & $2.160\times10^{-4}$ & $1.549\times10^{-4}$ & 436 & $1.534\times10^4$ & $4.363\times10^{-4}$\\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[!htb]
\centering\small
\setlength{\tabcolsep}{3.5pt}
\caption{Deterministic-reference refinement for two-dimensional variable scattering. The reference solver applies prescribed inflow according to the physical velocity direction. Discrepancies compare each level with its predecessor; reference refinement alone does not validate the random-feature boundary assembly.}
\label{tab:p5-reference}
\begin{tabular}{cccccccc}
\toprule
$\varepsilon$ & Level & Grid & $N_{\rm ang}$ & Iter. & Rel. residual & $\delta_f$ & $\delta_\rho$\\
\midrule
$1$ & A & $32^2$ & 8 & 244 & $9.88\times10^{-10}$ & - & -\\
 & B & $64^2$ & 16 & 536 & $9.86\times10^{-10}$ & $1.65\times10^{-3}$ & $1.58\times10^{-4}$\\
 & C & $96^2$ & 24 & 776 & $9.90\times10^{-10}$ & $6.32\times10^{-4}$ & $4.39\times10^{-5}$\\
\addlinespace
$10^{-3}$ & A & $32^2$ & 8 & 448 & $9.99\times10^{-10}$ & - & -\\
 & B & $64^2$ & 16 & 3470 & $9.98\times10^{-10}$ & $4.93\times10^{-5}$ & $4.93\times10^{-5}$\\
 & C & $96^2$ & 24 & 7880 & $1.00\times10^{-9}$ & $1.09\times10^{-5}$ & $1.09\times10^{-5}$\\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[!htb]
\centering\small
\setlength{\tabcolsep}{6pt}
\caption{OE-APRFM configuration for periodic transport with smooth initial data. MM uses the same total coefficient count and 4417 assembled rows. Component-wise periodic penalties differ from the objective obtained by directly penalizing $[f]$, which carries an $\varepsilon^2$ factor on $\|[j]\|^2$ on a symmetric angular domain.}
\label{tab:transient-config}
\begin{tabular}{@{}p{.32\linewidth}p{.63\linewidth}@{}}
\toprule
Quantity & Configuration\\
\midrule
Time interval and main step & $[0,0.2]$, $\Delta t=0.002$ (100 steps)\\
Scale parameters and seeds & $1,10^{-1},10^{-2},10^{-3}$; $11,23,37$\\
Features and precision & $J^r=J^j=128$, scale 1, double precision\\
Interior collocation & 128 spatial midpoints $\times$ 16 positive-angle Gauss points\\
Angular average & 64 Gauss points on the complete interval $[-1,1]$\\
Periodic traces & 32 positive-angle points for each of $r$ and $j$\\
Rows by block & $128+2048+2048+32+32=4288$\\
Independent evaluation & 512 spatial midpoints $\times$ 128 complete-angle Gauss points\\
Initial step & Analytic initial data enter the first-step right-hand side directly; no initial coefficient fit\\
Periodic weighting & Separate $r$ and $j$ jumps, each with positive-angle quadrature weights summing to one; each equation row is divided by its coefficient-vector $L^2$ norm\\
Algebraic cutoff & \texttt{rcond}$=10^{-12}$ after column $L^2$ equilibration; representative seed-11 ranks are 202 ($\varepsilon=1$) and 197 ($10^{-3}$) out of 256\\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[!htb]
\centering\small
\setlength{\tabcolsep}{4.5pt}
\caption{Backward-Euler refinement for periodic transport with seed 11 at $t=0.2$, measured against the continuous-time transport reference.}
\label{tab:time-refinement}
\begin{tabular}{cccccc}
\toprule
$\varepsilon$ & $\Delta t$ & $E_f^{\rm CT}$ & $E_\rho^{\rm CT}$ & $E_q^{\rm CT}$ & $p_f$\\
\midrule
$1$ & 0.004 & $8.826\times10^{-4}$ & $3.265\times10^{-4}$ & $9.897\times10^{-3}$ & -\\
 & 0.002 & $4.430\times10^{-4}$ & $1.653\times10^{-4}$ & $4.956\times10^{-3}$ & 0.9945\\
 & 0.001 & $2.219\times10^{-4}$ & $8.313\times10^{-5}$ & $2.480\times10^{-3}$ & 0.9973\\
\addlinespace
$10^{-3}$ & 0.004 & $7.042\times10^{-4}$ & $7.042\times10^{-4}$ & $6.922\times10^{-2}$ & -\\
 & 0.002 & $3.523\times10^{-4}$ & $3.523\times10^{-4}$ & $3.462\times10^{-2}$ & 0.9994\\
 & 0.001 & $1.762\times10^{-4}$ & $1.762\times10^{-4}$ & $1.732\times10^{-2}$ & 0.9998\\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[!htb]
\centering\small
\setlength{\tabcolsep}{4.5pt}
\caption{Periodic transport with seed 11 at $t=0.2$: discrepancies from the same-step BE transport reference and the BE diffusion density.}
\label{tab:be-refinement}
\begin{tabular}{cccccc}
\toprule
$\varepsilon$ & $\Delta t$ & $E_f^{\rm BE}$ & $E_\rho^{\rm BE}$ & $E_q^{\rm BE}$ & $E_{\rho,\mathrm{diff}}^{\rm BE}$\\
\midrule
$1$ & 0.004 & $4.375\times10^{-9}$ & $1.632\times10^{-9}$ & $2.740\times10^{-8}$ & $9.792\times10^{-2}$\\
 & 0.002 & $1.066\times10^{-8}$ & $5.665\times10^{-9}$ & $7.258\times10^{-8}$ & $9.844\times10^{-2}$\\
 & 0.001 & $4.306\times10^{-9}$ & $1.334\times10^{-9}$ & $2.735\times10^{-8}$ & $9.870\times10^{-2}$\\
\addlinespace
$10^{-3}$ & 0.004 & $1.529\times10^{-10}$ & $1.528\times10^{-10}$ & $3.003\times10^{-8}$ & $7.152\times10^{-8}$\\
 & 0.002 & $1.510\times10^{-10}$ & $1.509\times10^{-10}$ & $3.421\times10^{-8}$ & $6.744\times10^{-8}$\\
 & 0.001 & $1.373\times10^{-10}$ & $1.372\times10^{-10}$ & $3.609\times10^{-8}$ & $6.540\times10^{-8}$\\
\bottomrule
\end{tabular}
\end{table}

\begin{table}[!htb]
\centering\small
\setlength{\tabcolsep}{3.5pt}
\caption{Seed-11 same-grid field changes under feature or assembly-quadrature refinement. $r_0\to r_1$ lists the baseline and refined effective ranks.}
\label{tab:transient-field-checks}
\begin{tabular}{ccccccc}
\toprule
Check & $\varepsilon$ & $t$ & $r_0\to r_1$ & $\Delta f$ & $\Delta\rho$ & $\Delta q$\\
\midrule
Features & $1$ & 0.02 & $202\to237$ & $5.020\times10^{-10}$ & $4.451\times10^{-10}$ & $1.943\times10^{-8}$\\
 & & 0.10 & $202\to237$ & $2.751\times10^{-9}$ & $1.954\times10^{-9}$ & $3.163\times10^{-8}$\\
 & & 0.20 & $202\to237$ & $1.050\times10^{-8}$ & $5.609\times10^{-9}$ & $7.181\times10^{-8}$\\
 & $10^{-3}$ & 0.02 & $197\to235$ & $5.755\times10^{-10}$ & $5.731\times10^{-10}$ & $3.096\times10^{-8}$\\
 & & 0.10 & $197\to235$ & $2.187\times10^{-10}$ & $2.179\times10^{-10}$ & $3.099\times10^{-8}$\\
 & & 0.20 & $197\to235$ & $1.013\times10^{-10}$ & $1.012\times10^{-10}$ & $3.100\times10^{-8}$\\
\addlinespace
Quadrature & $1$ & 0.02 & $202\to202$ & $6.323\times10^{-13}$ & $5.291\times10^{-13}$ & $2.292\times10^{-11}$\\
 & & 0.10 & $202\to202$ & $5.444\times10^{-12}$ & $4.778\times10^{-12}$ & $2.884\times10^{-11}$\\
 & & 0.20 & $202\to202$ & $2.879\times10^{-11}$ & $2.175\times10^{-11}$ & $1.469\times10^{-10}$\\
 & $10^{-3}$ & 0.02 & $197\to197$ & $4.407\times10^{-13}$ & $4.402\times10^{-13}$ & $3.162\times10^{-12}$\\
 & & 0.10 & $197\to197$ & $1.110\times10^{-12}$ & $1.110\times10^{-12}$ & $8.896\times10^{-12}$\\
 & & 0.20 & $197\to197$ & $1.431\times10^{-12}$ & $1.431\times10^{-12}$ & $1.638\times10^{-11}$\\
\bottomrule
\end{tabular}
\end{table}
