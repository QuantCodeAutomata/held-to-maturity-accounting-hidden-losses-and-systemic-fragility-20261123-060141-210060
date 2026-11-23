"""Parameters for every experiment, taken from the paper where stated.

Each variable documents its meaning, unit, and source (a paper section,
table, or equation, or "not specified in the paper" when the paper gives
no value).
"""

# ----------------------------------------------------------------------
# Experiment 1: HTM holdings and asset return comovement (empirical study)
# ----------------------------------------------------------------------

# Twenty largest publicly traded U.S. bank holding companies, mapped to their
# lead national bank's FDIC certificate number for Call Report data and to
# their exchange ticker for stock returns. The paper names a subset of these
# tickers (BAC, BK, C, GS, JPM, MS, PNC, TFC, USB, WFC); the remaining ten are
# selected by total-asset rank among publicly traded bank holding companies,
# as the paper's "20 largest publicly traded U.S. banks" (p. 37) does not
# print a full ticker list.
BANKS_20 = {
    "JPM": 628, "BAC": 3510, "C": 7213, "WFC": 3511, "GS": 33124,
    "USB": 6548, "COF": 4297, "PNC": 6384, "TFC": 9846, "BK": 639,
    "MS": 32992, "STT": 14, "FITB": 6672, "HBAN": 6560, "SCHW": 57450,
    "MTB": 588, "KEY": 17534, "NTRS": 913, "RF": 12368, "ZION": 2270,
}
# First ten tickers of BANKS_20, the subset the paper names explicitly.
BANKS_10 = ["BAC", "BK", "C", "GS", "JPM", "MS", "PNC", "TFC", "USB", "WFC"]

# Sample period for stock returns and balance-sheet ratios (Sec. "Empirical
# Analysis", p. 37: "the 20 largest publicly traded U.S. banks from 2004 to
# 2023").
EXP1_STOCK_START = "2003-01-01"  # one extra year of bars so 2004 daily returns can be computed
EXP1_SAMPLE_START = "2004-01-01"  # paper's sample start, p. 37
EXP1_SAMPLE_END = "2023-12-31"  # paper's sample end, p. 37

# Bank-count / asset-concentration table: FFIEC/FDIC bank counts 2001-2023
# (p. 37, "data_requirements"). Annual snapshots at fiscal year-end.
EXP1_COUNT_START_YEAR = 2001
EXP1_COUNT_END_YEAR = 2023

# Top-N bank groups used for the balance-sheet ratio tables (p. 37).
EXP1_TOP_N_GROUPS = (4, 20, 50)
EXP1_RATIO_START_YEAR = 2004
EXP1_RATIO_END_YEAR = 2023

# Pre/post-COVID windows for the pairwise correlation table (p. 11, Table 3).
# The paper distinguishes "before the pandemic" and "after the pandemic"
# (Table 3 note) but does not give exact cutoff dates: not specified in the
# paper. The WHO declared COVID-19 a pandemic on 2020-03-11, so the pre-COVID
# window is the three full years before that date and the post-COVID window
# is the remainder of the sample through the paper's 2023 sample end.
EXP1_PRE_COVID_START = "2017-01-01"
EXP1_PRE_COVID_END = "2020-02-29"
EXP1_POST_COVID_START = "2020-03-01"
EXP1_POST_COVID_END = "2023-12-31"
EXP1_PAIRWISE_BANKS = BANKS_10  # paper reports this table for 10 banks, p. 11

# Rolling window (trading days) used to build the quarterly asset-weighted
# correlation series feeding the two OLS regressions (p. 37-39). Not given a
# specific length in the paper: one trailing trading year is used, the
# standard window for daily-return correlation estimation.
EXP1_AW_CORR_WINDOW_DAYS = 252  # not specified in the paper; one trading year

# Download-time budget: the FDIC BankFind Suite API serves one HTTP request
# per quarter per top-N group (`EXP1_TOP_N_GROUPS`) plus one per calendar
# year for the bank-count table, well under the two-hour guideline, so no
# universe reduction is needed for Experiment 1.

# ----------------------------------------------------------------------
# Agent-based interbank model shared by Experiments 2 and 3
# (Sec. "Agent-Based Model", pp. 18-29; Table 4 "Calibrated Parameters")
# ----------------------------------------------------------------------

N_BANKS = 100  # number of banks in the system, Table 4
LEVERAGE_MAX = 10.0  # leverage ratio upper bound (Assets / Equity), Table 4
CAPITAL_RESERVE_RATIO = 0.20  # capital reserve as a ratio of deposits, Table 4
FED_FUNDS_RATE = 0.04  # interbank loan interest rate, Table 4
PORTFOLIO_MU = 0.10  # portfolio (risky-asset) expected annual return, Table 4
PORTFOLIO_SIGMA = 0.18  # portfolio return annual volatility, Table 4
KAPPA_L = 0.90  # interbank-claim recovery rate in default, Table 4
KAPPA_E = 0.50  # external-portfolio recovery rate in default, Table 4

# The paper's Eq. 7 (R = mu + LZ) gives mu and sigma without stating the
# length of one simulation episode. Portfolio Return Rate and Volatility
# (Table 4) are standard annualized return/volatility figures, so each
# episode is treated as one trading day (mu, sigma scaled by 1/N and
# 1/sqrt(N) with N=EPISODES_PER_YEAR) so that a 100-bank system can survive
# the 300 pre-shock episodes described on p. 25 before the shock is applied;
# not specified in the paper.
EPISODES_PER_YEAR = 252

# Maximum number of borrowing requests a bank issues per episode. Table 4
# ("Hyperparameter of the base model") lists a general maximum of 10
# borrowing requests, but Table 7 ("Parameter of the Experiments") lists 5
# for the correlation and shock-timing experiments reproduced here; the
# experiment-specific value is used.
N_BORROW_REQUESTS = 5  # Table 7

# Size of each borrowing request, as a ratio of the borrower's equity that
# period (Table 4 lists "Size of the borrow as a ratio of Equity" = 1.00).
# Interpreted as the *total* amount a bank seeks to borrow in one episode
# (equal to its own equity), split evenly across its N_BORROW_REQUESTS
# requests, each capped so the bank's own leverage constraint is respected.
BORROW_SIZE_RATIO_OF_EQUITY = 1.00  # Table 4

# Dirichlet-categorical counterparty selection and belief updating
# (Sec. "Network Formation", Eq. 5-6). The memory-decay parameter Q that
# down-weights past observations is not given a numerical value in the
# paper ("not specified in the paper"); Q=0.95 (geometric decay of past
# approval counts) is used so that the stated "memory decay mechanism to
# prevent lock-in" has an effect, instead of Q=1 which would remove it.
DIRICHLET_PRIOR_CONCENTRATION = 1.0  # symmetric initial Dirichlet belief, Eq. 5
BELIEF_MEMORY_DECAY_Q = 0.95  # not specified in the paper; assumed decay rate

# Episodes (time steps) simulated before the systemic shock is realized.
# The paper applies the shock "at t=300" (pp. 25-29); 300 pre-shock episodes
# are simulated so the endogenous network has matured before the shock.
PRE_SHOCK_EPISODES = 300

# Shock magnitude and design (Sec. "Systemic Shock", Eq. 8, p. 25): all banks
# receive a simultaneous adverse return R = -|Delta_sigma * L Z|.
SHOCK_DELTA_SIGMA = 5.00  # volatility spike multiplier applied to the shock, Table 4/7

# Experiment 2: correlation and nonlinear systemic risk (p. 4, p. 27). A
# single shock at t=300 under three asset-return correlation regimes.
EXP2_CORRELATIONS = (0.0, 0.8, 0.9)  # rho values compared, p. 27
EXP2_SHOCK_TIMING = (1,)  # a single shock episode (t=300): "Base Model" timing

# Experiment 3: timing of loss recognition, HTM (concentrated) vs. MTM
# (gradual) accounting (p. 5, p. 29). Correlation fixed at the Experiment-2
# base value (rho=0.80); total shock magnitude held constant while spread
# over 1, 2 or 3 shock episodes ending at t=300.
EXP3_RHO = 0.80  # fixed correlation for the timing comparison, p. 29
EXP3_SHOCK_TIMINGS = (1, 2, 3)  # number of episodes the total shock is split across

# Monte Carlo repetitions. The paper runs 2000 independent simulations per
# configuration (p. 27, p. 29). One simulated run of the 100-bank, 300-episode
# model takes a fraction of a second, so the full paper count across all six
# configurations (3 in Experiment 2, 3 in Experiment 3) completes well inside
# the time available for this replication; the results reported in
# results/RESULTS.md use `--full`, i.e. N_RUNS_PAPER. N_RUNS_RUN is kept as a
# fast default (no `--full` flag) for quick smoke checks of the experiment
# scripts, not for the reported results.
N_RUNS_PAPER = 2000  # Monte Carlo repetitions per configuration, p. 27/29
N_RUNS_RUN = 150  # fast default for a quick smoke check, not the reported results

RANDOM_SEED = 42  # seeds every experiment for reproducibility; not specified in the paper
