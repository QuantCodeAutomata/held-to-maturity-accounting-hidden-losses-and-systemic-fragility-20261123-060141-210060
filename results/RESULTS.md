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

| Regressor | Slope | Slope s.e. | t-stat | p-value | R² | Paper slope (Table 9) | Paper R² |
|---|---|---|---|---|---|---|---|
| Securities Ratio | 0.472 | 0.138 | 3.43 | 0.0011 | 0.166 | 0.530 | 0.204 |
| HTM Ratio | 0.849 | 0.181 | 4.69 | 1.7e-05 | 0.271 | 0.142 | 0.260 |

**Verdict: Securities Ratio regression partially reproduced, HTM Ratio
regression partially reproduced.** Both slopes are positive and
statistically significant at the 1% level, reproducing the paper's
qualitative claim (Table 9) that the Securities Ratio and the HTM Ratio both
positively and significantly predict the level of asset-weighted return
correlation among large banks. The Securities Ratio slope (0.472 vs. the
paper's 0.530, a 10.9% difference) and its R² (0.166 vs. 0.204) are close to
the paper's values. The HTM Ratio slope (0.849) overshoots the paper's 0.142
by a factor of about six, a material difference in magnitude, even though
its R² (0.271 vs. 0.260) matches closely. Since the slope of a simple OLS
regression equals $\mathrm{Cov}(x,y)/\mathrm{Var}(x)$, a matching R² with a
differing slope implies the two samples' asset-weighted HTM ratio have a
different variance; the cause of that variance difference between this
replication's 2004-2023 regulatory-filing panel and the paper's underlying
sample is not identified.

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
paper's reported mean defaults.

A second check isolates the correlation-generation step itself from the
network/clearing mechanism: Eq. 7's one-factor construction
$Z=\sqrt\rho\,\text{common}+\sqrt{1-\rho}\,\text{idiosyncratic}$ is verified on
2·10^5 synthetic draws against the known folded-normal property that
$\mathrm{Corr}(|Z_i|,|Z_j|)$ and the probability of a joint tail event both
rise monotonically, but smoothly, in $\rho$ (e.g. $P(|Z_i|>3\sigma\text{ and
}|Z_j|>3\sigma\mid\text{marginal}>3\sigma)$ = 0.241 at $\rho=0.8$ versus 0.445
at $\rho=0.9$, an 85% relative rise, not the ≈5x rise in mean defaults the
paper reports from $\rho=0.8$ to $\rho=0.9$). This confirms the correlated-draw
code (Eq. 7) is implemented correctly and is not the source of the gap: a
smooth, well-behaved increase in joint-tail probability is exactly what it
produces.

A third, algebraic check identifies the specific mechanism that caps the
first-round (direct, own-portfolio) default rate's dependence on $\rho$: by
construction, $Z_i=\sqrt{\rho}\,\text{common}+\sqrt{1-\rho}\,\text{idio}$ has
$\mathrm{Var}(Z_i)=\rho+(1-\rho)=1$ for every $\rho$ (Eq. 7), so each bank's
own marginal shock distribution — and hence the probability that its own
portfolio loss alone breaches its equity buffer — is exactly invariant to
$\rho$. Raising $\rho$ therefore cannot raise the *expected number* of
directly-shocked defaults; it can only redistribute the same total
probability mass toward more simultaneous extreme draws, which is exactly
what the run shows (std rising from 2.8 at $\rho=0$ to 19.3/22.8 at
$\rho=0.8/0.9$, and a full 100/100 cascade appearing in 6.3%/7.2% of runs at
$\rho=0.8/0.9$ versus never at $\rho=0$). Whether this redistributed tail mass
shows up in the *mean* default count instead depends entirely on the
interbank network's second-round contagion: a bank that fails from its own
shock must also drag down solvent counterparties through unpaid interbank
claims for the mean to rise sharply with $\rho$. In this implementation,
interbank borrowing is sized to cover only a bank's capital-reserve shortfall
($\mathrm{BORROW\_SIZE\_RATIO\_OF\_EQUITY}=1.00$ against the shortfall,
Table 4), which keeps interbank claims a small fraction of each bank's
balance sheet, so second-round contagion is weak relative to the first-round,
correlation-invariant direct defaults; this limits the $\rho$-driven
amplification of the *mean* default count to what is observed (8.09 → 8.46 →
8.66) even though the tail statistics respond to $\rho$ exactly as the
paper's mechanism predicts. The paper's substantially larger $\rho=0.9$ jump
in the mean therefore requires either a stronger second-round contagion
channel than Table 4's capital-reserve-shortfall borrowing rule produces, or
network/leverage parameters the paper leaves unspecified (the belief
memory-decay $Q$ in Eq. 6; the exact Rogers & Veraart (2013) iteration order);
the precise combination that reproduces the paper's $\rho=0.9$ tipping point
is not identified.

A further check tested whether simply strengthening this channel — sizing the
borrowing request to the bank's full equity (Table 4's literal ratio applied
to equity rather than to the capital-reserve shortfall) and investing the
borrowed proceeds in the risky Portfolio so that leverage can build up
episode over episode — reproduces the paper's $\rho=0.9$ jump. On a reduced
run this instead drove leverage to the regulatory bound in nearly every
episode and produced near-universal default (95% full-cascade runs) at every
$\rho$, including $\rho=0$, destroying the correlation dependence the
experiment measures rather than reproducing it; the change was reverted. This
rules out "borrowing proportional to equity, fully re-invested every episode"
as the missing mechanism and confirms that reproducing the paper's exact
tipping point requires a network/leverage calibration this implementation
could not recover from the text alone.

Plot: `exp2_defaults_by_correlation.png`.

## Experiment 3 — Agent-Based Model: Timing of Loss Recognition (HTM vs MTM)

File: `exp3_shock_timing_defaults.csv`, same `N_RUNS_PAPER = 2000`, ρ=0.80,
total shock magnitude held constant across 1, 2, and 3 shock episodes ending
at t=300 (Table 7, p. 26).

| Shock episodes | Mean defaults (this run) | Mean defaults (paper, p. 29) |
|---|---|---|
| 1 (HTM analogue) | 8.46 | ≈12.03 |
| 2 | 8.12 | ≈1.68 |
| 3 (MTM analogue) | 7.95 | ≈0.08 |

Plot: `exp3_defaults_by_shock_timing.png`.

**Verdict: partially reproduced.** The qualitative claim of the experiment —
splitting the same total loss into more, smaller episodes lowers defaults
versus concentrating it into one — holds: mean defaults fall monotonically
from 8.46 (1 shock) to 8.12 (2 shocks) to 7.95 (3 shocks), confirming the sign
of the HTM-vs-MTM effect. The magnitude is not reproduced: the paper reports a
collapse to near zero by 3 episodes (≈0.08), while this run only shows a
modest decline (-6% from 1 to 3 episodes). The 1-shock row repeats
Experiment 2's ρ=0.8 result because the two configurations are numerically
identical.

The per-episode shock scale is normalized so that the *expected total*
one-sided loss, not its variance, stays constant across different numbers of
shock episodes: since each shock is $R=-|\Delta_\sigma\sigma Z|$ (Eq. 8), the
sum of $k$ such draws scales linearly in $k$ for a fixed per-episode
$\Delta_\sigma$, so each episode's $\Delta_\sigma$ is divided by $k$ (not by
$\sqrt{k}$, which would be the right normalization only for a zero-mean sum).
Under this normalization the experiment's directional claim holds, but the
remaining gap to the paper's reported near-zero default count for 3 episodes
is not explained by this normalization choice alone. As in Experiment 2, the
synthetic check of Eq. 7's correlated-draw construction rules out the shock
generator itself as the cause of the remaining gap. The same mechanism
identified in Experiment 2 explains why the decline here is so much milder
than the paper's: the direct, first-round default rate from a bank's own
portfolio shock depends on $\rho$ only through how often banks are hit
*simultaneously*, not on the per-episode shock size itself staying large, so
splitting $\Delta_\sigma$ across more, smaller episodes mainly thins the
right tail of simultaneous extreme draws rather than collapsing the
first-round default rate to zero; and because interbank exposure in this
calibration is sized only to each bank's capital-reserve shortfall (Table 4),
weak second-round contagion cannot turn a modest first-round decline into the
paper's near-total collapse by 3 episodes. The precise parameter combination
that would reproduce the paper's steep decline is not identified.

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
- **Splitting a fixed total shock across episodes (Experiment 3).** The paper
  states only that "the same total shock magnitude" is spread over 1, 2 or 3
  episodes (p. 29) without giving the per-episode scaling formula. Since each
  episode's loss is one-sided ($R=-|\Delta_\sigma\sigma Z|$, Eq. 8), this
  implementation divides $\Delta_\sigma$ by the number of episodes $k$ so the
  *expected total* loss across the $k$ episodes matches the single-episode
  case; this is an interpretation of a detail the paper leaves open, not a
  formula the paper gives.
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

- **HTM Ratio regression slope (Table 9).** The asset-weighted HTM Ratio
  slope (0.849) is about six times the paper's reported 0.142, even though
  the regression's R² matches closely (0.271 vs. 0.260) and the sign and
  1%-level significance reproduce the paper's claim. Because a simple OLS
  slope scales inversely with the regressor's variance for a fixed R², this
  points to a difference in the variance of the asset-weighted HTM Ratio
  between this replication's panel and the paper's; the specific cause of
  that variance difference is not identified.

## Not reproduced

- The paper's Table 8 and the accompanying Figure/Table for Experiment 3 are
  only partially reproduced: see the Experiment 3 table above. The direction
  of the effect (fewer defaults as the same total loss is split over more
  episodes) is reproduced, but the mean-default levels and the size of the
  drop across 1/2/3 episodes are not matched, for the reasons given above
  (unresolved episode time-scale of the shock equation, and the unspecified
  per-episode scaling formula for a fixed total shock).
- Figure 7 (network centrality / intra-financial system assets and
  liabilities evolution under different network topologies) and Figure 8
  (illustrative HTM-vs-MTM asset-value diagram) are not reproduced as
  separate artifacts: they illustrate the same mechanism already quantified in
  the Experiment 2 and Experiment 3 tables above, and the paper gives no
  numerical values for Figure 7's "Learning/Ring/Full Connection Model" curves
  beyond the sparse table already used as a qualitative check during
  development.
