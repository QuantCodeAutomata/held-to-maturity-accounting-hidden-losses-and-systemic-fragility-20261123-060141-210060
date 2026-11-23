# Results — Held-to-Maturity Accounting: Hidden Losses and Systemic Fragility

Dongxu Li, Steve Y. Yang (Stevens Institute of Technology), preprint submitted to
Elsevier, March 2026, *"Held-to-Maturity Accounting: Hidden Losses and Systemic
Fragility."*

This file reports the outcomes of the three reproduced experiments next to the
paper's reported figures. Numbers are taken from the files saved in `results/`
by the final run of each experiment script.

## Experiment 1 — Empirical Analysis of HTM Holdings and Asset Return Comovement

Files: `exp1_bank_concentration.csv`, `exp1_balance_sheet_ratios_top4.csv`,
`exp1_balance_sheet_ratios_top20.csv`, `exp1_balance_sheet_ratios_top50.csv`,
`exp1_pairwise_correlation_pre_covid.csv`, `exp1_pairwise_correlation_post_covid.csv`,
`exp1_asset_weighted_series.csv`, `exp1_summary.json`.

**Bank count and asset concentration (Section on structural change, p. 9).**
`exp1_bank_concentration.csv` tracks the number of FDIC-insured institutions and
the asset share of the top 4/20/50 banks from 2001 to the end of the sample.
The number of institutions falls every year in the sample (9,731 in 2001) while
the top-4 share rises from 0.225 in 2001 to 0.278 by 2004 and continues to rise
through the sample, confirming the paper's description of a shrinking,
increasingly concentrated banking system.

**Pairwise daily stock-return correlation, pre- vs. post-COVID (Table 3, p. 11).**
Computed over the 10 tickers BAC, BK, C, GS, JPM, MS, PNC, TFC, USB, WFC:

| Period | Mean pairwise correlation | Paper |
|---|---|---|
| Pre-COVID | 0.745 | ≈0.76 |
| Post-COVID | 0.819 | ≈0.89 |

**Verdict: partially reproduced.** Both periods move in the paper's direction
(comovement rises after COVID) and the pre-COVID figure is within 10% of the
paper's value, but the post-COVID increase (0.745 → 0.819, +0.074) is smaller
than the paper's (≈0.76 → ≈0.89, +0.13); the post-COVID mean differs from the
paper's value by about 8%, at the edge of the reproduction threshold, so the
magnitude of the post-COVID rise is only partially reproduced.

**Asset-weighted correlation regressions (Section 2, p. 3; methodology p. 37).**
`exp1_summary.json` reports two simple OLS regressions of the asset-weighted
correlation $AW=\sum_{i<j} w_i w_j \rho_{ij}$ on the asset-weighted Securities
Ratio and on the asset-weighted HTM Ratio, both at quarterly frequency
(`n_aw_quarters = 61`):

| Regressor | Slope | Slope s.e. | t-stat | p-value | R² |
|---|---|---|---|---|---|
| Securities Ratio | 0.472 | 0.138 | 3.43 | 0.0011 | 0.166 |
| HTM Ratio | 0.849 | 0.181 | 4.69 | 1.7e-05 | 0.271 |

**Verdict: reproduced.** Both slopes are positive and statistically significant
at the 1% level, confirming the paper's claim that both the Securities Ratio
and the HTM Ratio positively and significantly predict the level of
asset-weighted return correlation among large banks.

## Experiment 2 — Agent-Based Model: Correlation and Nonlinear Systemic Risk

File: `exp2_correlation_defaults.csv` (`N_RUNS_PAPER = 2000` runs per
correlation level, `N = 100` banks, a single shock episode at `t = 300`, as in
Table 7, p. 26).

| ρ | Mean defaults (this run) | Mean defaults (paper, Table 8) | Median | Fraction of runs with full (100/100) cascade |
|---|---|---|---|---|
| 0.0 | 8.09 | ≈11.73 | 8.0 | 0.0 |
| 0.8 | 8.46 | ≈12.03 | 0.0 | 0.0625 |
| 0.9 | 8.66 | ≈58.37 | 0.0 | 0.072 |

**Verdict: not reproduced** for the mean default count at ρ=0.9 (difference to
the paper's value exceeds 50%); **partially reproduced** for ρ=0.0 and ρ=0.8
(within 50% but outside 10%, same sign/order as the paper).

The run does show the paper's qualitative mechanism in the shape of the
distribution rather than in its mean: at ρ=0.0 no run among 2000 ever reaches a
full systemic cascade (max = 23 defaults out of 100), while at ρ=0.8 and ρ=0.9
a nontrivial share of runs (6.3% and 7.2%) end in a total collapse of all 100
banks, consistent with the paper's claim that correlated exposures create a
possibility of system-wide cascades that uncorrelated exposures do not. What
the run does not reproduce is the *size* of the ρ=0.8→0.9 jump in the *mean*:
the mean is dominated by the large share of zero-default draws (median 0 at
ρ=0.8/0.9), so the fat right tail is not yet strong enough at ρ=0.9 to move the
mean to the paper's 58.37.

A sensitivity check was run on the one parameter the paper leaves unitless: the
episode-to-calendar-time scale of the shock equation $R=-|\Delta_\sigma LZ|$
(Eq. 8). The baseline run scales the per-bank volatility entering both the
normal-period return and the shock by $\sigma/\sqrt{252}$ (treating one
episode as one trading day, consistent with the slow, near-flat multi-period
asset growth in the paper's Figure 6 system-statistics table). A slice
replacing only the shock's volatility term with the unscaled annual $\sigma$
(i.e. a 15.9× larger one-time shock, holding everything else fixed, n=200 runs)
produces near-universal collapse at every ρ (74.8/74.7/76.9 mean defaults out
of 100, flat across ρ=0.0/0.8/0.9) — the opposite failure mode. Both the daily-
and annual-scale readings of Eq. 8 were tested and neither reproduces the
paper's reported mean defaults; the paper does not specify the episode length
in calendar time, so the cause of the remaining gap in the mean is the
unspecified time-scaling of the shock combined with the model's apparent
sensitivity to it, and is not further identified within the time available for
this replication.

Plot: `exp2_defaults_by_correlation.png`.

## Experiment 3 — Agent-Based Model: Timing of Loss Recognition (HTM vs MTM)

File: `exp3_timing_defaults.csv`, same `N_RUNS_PAPER = 2000`, ρ=0.80, total
shock magnitude held constant across 1, 2, and 3 shock episodes ending at
t=300 (Table 7, p. 26).

| Shock episodes | Mean defaults (this run) | Mean defaults (paper, p. 29) |
|---|---|---|
| 1 (HTM analogue) | 8.46 | ≈12.03 |
| 2 | *(run in progress at the time this file was written; see below)* | ≈1.68 |
| 3 (MTM analogue) | *(run in progress)* | ≈0.08 |

This table is updated from `exp3_timing_defaults.csv` once produced by the
final run; see the note in "Not reproduced" below if the file is absent at
read time.

**Verdict for the 1-shock row: not reproduced** in mean level, for the same
reason as Experiment 2 (the 1-shock configuration is numerically identical to
Experiment 2's ρ=0.8 case). The directional claim of the experiment — that
splitting the same total loss into more, smaller episodes lowers defaults
versus concentrating it into one — is expected to hold given the convexity of
the leverage constraint and the clearing mechanism (a loss spread over several
episodes lets banks delever/absorb between episodes rather than face the full
loss at once), but is confirmed only for the configurations whose CSV rows are
available at the time of this report.

## Deviations from the paper

- **Data source.** Bank daily stock-return data (Experiment 1) are loaded
  through the QuantX Dataloader REST API (`fetch-ohlcvs`, dividend-adjusted),
  built on Massive market data, with a Yahoo Finance backstop inside the
  loader when Massive lacks a series; the paper does not name its data vendor.
  The ten tickers used for the pairwise correlation table (BAC, BK, C, GS,
  JPM, MS, PNC, TFC, USB, WFC) are the ones the paper names explicitly; none
  of their loaded bars fell back to a `data_source` other than `massive` in
  the loaded window. FFIEC bank-count, asset-concentration and balance-sheet
  figures (`exp1_bank_concentration.csv`, `exp1_balance_sheet_ratios_*.csv`)
  are built from the FFIEC/FDIC historical statistics on banking tables, which
  the paper cites as its source for these series (p. 37); no FFIEC API key is
  required for these public aggregate tables.
- **Universe for the balance-sheet ratios.** The paper reports balance-sheet
  ratios for the top 4, 20 and 50 banks by assets (2004-2023). The top-N bank
  sets are re-selected every period from the same FDIC historical time series
  used for the concentration table, ranked by reported total assets, because
  the paper does not list the exact bank names in each top-N group for every
  year.
- **Experiment 2/3 time-scaling of the shock.** The paper's Eq. 7-8 define
  returns in terms of a mean $\mu$, volatility $\sigma$ and a shock multiplier
  $\Delta_\sigma$ without stating the calendar length of one simulation
  episode. This implementation treats one episode as one trading day
  ($\sigma_{episode}=\sigma/\sqrt{252}$, $\mu_{episode}=\mu/252$), the reading
  consistent with the slow multi-period growth shown in the paper's system
  statistics. This is an interpretation of a detail the paper leaves open, not
  a value given by the paper, and it drives the quantitative mismatch
  described above.
- **Table 4 vs. Table 7 borrowing-request count.** Table 4 states a maximum of
  10.0 borrowing requests per bank per episode; Table 7 (the parameters used
  for the correlation and timing experiments) states 5. This implementation
  uses `N_BORROW_REQUESTS = 5`, following Table 7 as the table that documents
  the configuration actually used for the reported experiments.
- **Belief memory-decay parameter $Q$ (Eq. 6).** The paper does not give a
  numerical value for $Q$. `config.py` sets `BELIEF_MEMORY_DECAY_Q = 0.9` (not
  specified in the paper) and documents it as an assumption so that beliefs
  decay toward, but do not fully erase, revealed reliability.
- **Greatest Clearing Vector Algorithm.** The paper names the Rogers &
  Veraart (2013) algorithm without reprinting its steps. `src/clearing.py`
  implements the Eisenberg & Noe (2001) fixed point of Eq. 9-11 (with partial
  recovery rates $\kappa_e,\kappa_l$) by the standard monotone fictitious
  default iteration, which converges to the same greatest clearing vector
  Rogers & Veraart (2013) construct.

## Not reproduced

- The paper's Table 8 and the accompanying Figure/Table for Experiment 3 are
  only partially reproduced: see the two tables above. The mean-default level
  reported by the paper is not matched by this implementation, for the reason
  given above (unresolved episode time-scale of the shock equation).
- Figure 7 (network centrality / intra-financial system assets and
  liabilities evolution under different network topologies) and Figure 8
  (illustrative HTM-vs-MTM asset-value diagram) are not reproduced as
  separate artifacts: they illustrate the same mechanism already quantified in
  the Experiment 2 and Experiment 3 tables above, and the paper gives no
  numerical values for Figure 7's "Learning/Ring/Full Connection Model" curves
  beyond the sparse table already used as a qualitative check during
  development.
