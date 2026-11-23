# Held-to-Maturity Accounting: Hidden Losses and Systemic Fragility

Reproduction of selected results from *Held-to-Maturity Accounting: Hidden
Losses and Systemic Fragility*.

This repository reproduces three experiments described in the paper:

1. **Experiment 1 — Empirical analysis of HTM holdings and asset return
   comovement** (`exp/exp_1_htm_holdings_and_comovement.py`): an empirical
   study of U.S. bank balance sheets and stock returns that documents the
   growth in Held-To-Maturity (HTM) securities holdings and asset
   concentration, the pre- vs. post-COVID rise in pairwise bank stock-return
   correlation, and tests whether an asset-weighted Securities Ratio and HTM
   Ratio positively predict an asset-weighted correlation measure.
2. **Experiment 2 — Agent-based model: correlation and nonlinear systemic
   risk** (`exp/exp_2_correlation_and_systemic_risk.py`): a dynamic
   agent-based interbank model in which a single systemic shock is applied
   under three constant asset-return correlations (rho = 0.0, 0.8, 0.9),
   measuring the average number of bank defaults.
3. **Experiment 3 — Agent-based model: timing of loss recognition (HTM vs.
   MTM)** (`exp/exp_3_timing_of_loss_recognition.py`): the same interbank
   model with the same total shock magnitude spread over 1, 2 or 3 shock
   episodes, mimicking concentrated (HTM) vs. gradual (MTM) loss
   recognition.

## Repository layout

```
src/            shared library: config, empirical analysis, ABM, clearing
exp/            experiment entry points (python -m exp.<name>)
data/           data loading: FDIC/FFIEC balance sheets, stock OHLCV bars
tests/          pytest unit and (optional) market-data tests
results/        tables, metrics and plots written by the experiments
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in DATALOADER_API_KEY and MASSIVE_API_KEY for Experiment 1
```

Experiment 1 needs network access to the QuantX Dataloader (daily bank stock
bars) and to the public FDIC BankFind Suite API (balance-sheet data, no key
required). Experiments 2 and 3 are self-contained simulations and need no
API keys or network access.

## Running the experiments

```bash
python -m exp.exp_1_htm_holdings_and_comovement
python -m exp.exp_2_correlation_and_systemic_risk --full
python -m exp.exp_3_timing_of_loss_recognition --full
```

Without `--full`, Experiments 2 and 3 run a reduced number of Monte Carlo
repetitions per configuration (`src.config.N_RUNS_RUN`) for a quick smoke
check; `--full` runs the paper's 2000 repetitions per configuration
(`src.config.N_RUNS_PAPER`), which is the configuration used for the results
reported in `results/RESULTS.md`.

All three scripts write their tables, metrics (JSON) and plots directly to
`results/`, overwriting any previous run's files there.

## Tests

```bash
python -m pytest -m "not market_data"   # unit tests on synthetic/small fixtures
python -m pytest                         # also runs the market-data test (needs DATALOADER_API_KEY)
```

## Reproduce everything

```bash
./reproduce.sh
```

runs the test suite and all three experiments in the configuration used for
`results/RESULTS.md`.

See `results/RESULTS.md` for the comparison against the paper's reported
values, deviations from the paper, and what is not reproduced.
