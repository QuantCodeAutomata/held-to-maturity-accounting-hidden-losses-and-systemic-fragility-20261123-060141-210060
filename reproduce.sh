#!/usr/bin/env bash
# Runs the test suite and every experiment in the configuration used for
# results/RESULTS.md (Experiments 2 and 3 with --full, i.e. the paper's 2000
# Monte Carlo repetitions per configuration).
set -uo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"

echo "== Unit tests (excluding market-data tests) =="
python -m pytest -m "not market_data" -q || exit 1

echo "== Experiment 1: HTM holdings and asset-return comovement =="
python -m exp.exp_1_htm_holdings_and_comovement || exit 1

echo "== Experiment 2: correlation and nonlinear systemic risk (full) =="
python -m exp.exp_2_correlation_and_systemic_risk --full || exit 1

echo "== Experiment 3: timing of loss recognition (full) =="
python -m exp.exp_3_timing_of_loss_recognition --full || exit 1

echo "Done. See results/ for tables, metrics and plots."
