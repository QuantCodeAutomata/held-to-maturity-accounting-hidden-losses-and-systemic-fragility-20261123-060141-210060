# AGENTS.md

Guide for coding agents and contributors working in this repository.

## What this repository implements

A reproduction of three experiments from *Held-to-Maturity Accounting: Hidden
Losses and Systemic Fragility*:

1. An empirical study of U.S. bank balance sheets (HTM securities holdings,
   asset concentration) and daily stock-return comovement across large U.S.
   banks, with two OLS regressions of an asset-weighted correlation measure
   on the asset-weighted Securities Ratio and HTM Ratio.
2. An agent-based interbank model (endogenous counterparty selection,
   correlated risky-asset returns, Eisenberg & Noe / Rogers & Veraart
   clearing) simulating a single systemic shock under three asset-return
   correlations (rho = 0.0, 0.8, 0.9).
3. The same agent-based model with the same total shock magnitude spread
   over 1, 2 or 3 shock episodes, contrasting concentrated (HTM-like) and
   gradual (MTM-like) loss recognition.

## Layout

```
src/        flat library modules: config.py, empirical.py, abm.py, clearing.py
exp/        one script per experiment, run as `python -m exp.<name>`
data/       data_loader.py (QuantX Dataloader + FDIC) — no synthetic generator needed
tests/      pytest unit tests (synthetic/small fixtures) and one market-data test
results/    tables (csv/json), plots (png) and RESULTS.md written by the experiments
```

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in DATALOADER_API_KEY and MASSIVE_API_KEY, Experiment 1 only

python -m pytest -m "not market_data"   # default test run, no network access
python -m pytest                        # also exercises data/data_loader.py (needs the keys)

python -m exp.exp_1_htm_holdings_and_comovement
python -m exp.exp_2_correlation_and_systemic_risk --full
python -m exp.exp_3_timing_of_loss_recognition --full

./reproduce.sh   # tests + all three experiments in the configuration used for RESULTS.md
```

## Conventions

- All experiment parameters live in `src/config.py` as module-level
  `UPPER_CASE` variables, each with a comment naming its meaning, unit and
  source (a paper table/equation, or "not specified in the paper"). No
  classes or parameter dicts.
- Market data is loaded only through `data/data_loader.py` (QuantX Dataloader
  `fetch-ohlcvs`, and the public FDIC BankFind Suite API for balance-sheet
  series). No experiment script or `src/` module fetches data on its own.
- `exp/` scripts call the public functions of `src/empirical.py` and
  `src/abm.py` with their documented argument names and return keys; they do
  not define private data loaders or alternative metric definitions.
- `DATALOADER_API_KEY` and `MASSIVE_API_KEY` are read from the environment
  (via `.env`, never committed) and used only in requests to the QuantX and
  Massive APIs.
- Running an experiment script overwrites its files in `results/`. The
  numbers and verdicts in `results/RESULTS.md` are not regenerated
  automatically; after a rerun, check them against the new files in
  `results/` and update them by hand.
- Tests: unit tests use synthetic inputs or small fixtures; the one
  market-data test is marked `market_data` and is skipped automatically when
  `DATALOADER_API_KEY` is absent. Run `python -m pytest -m "not market_data"`
  for routine changes and the full suite, including the market-data test,
  before committing a change to the data path.
