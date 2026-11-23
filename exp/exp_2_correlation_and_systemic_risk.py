"""Experiment 2: asset-return correlation and nonlinear systemic risk.

Runs the agent-based interbank model (`src.abm.run_simulation`) under a
single systemic shock at the end of the pre-shock period, for three
constant pairwise asset-return correlations rho in {0.0, 0.8, 0.9}
(paper, pp. 4, 25-27), and records the number of bank defaults per
Monte Carlo run.

Usage
-----
    python -m exp.exp_2_correlation_and_systemic_risk [--full]

Without ``--full`` the executed number of Monte Carlo repetitions is
``src.config.N_RUNS_RUN``; with ``--full`` it is the paper's
``src.config.N_RUNS_PAPER`` (2000 runs per configuration, p. 27).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import config
from src.abm import run_simulation

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run_experiment(n_runs: int) -> pd.DataFrame:
    """Run the model at each correlation in `config.EXP2_CORRELATIONS`.

    Returns one row per correlation with summary statistics of the number
    of defaults over `n_runs` Monte Carlo repetitions.
    """
    rows = []
    for rho in config.EXP2_CORRELATIONS:
        start = time.time()
        result = run_simulation(
            n_runs=n_runs,
            rho=rho,
            shock_episodes=config.EXP2_SHOCK_TIMING,
            n_banks=config.N_BANKS,
            pre_shock_episodes=config.PRE_SHOCK_EPISODES,
            delta_sigma=config.SHOCK_DELTA_SIGMA,
            seed=config.RANDOM_SEED,
        )
        elapsed = time.time() - start
        defaults = result.n_defaults
        rows.append({
            "rho": rho,
            "n_runs": n_runs,
            "n_banks": config.N_BANKS,
            "mean_defaults": float(defaults.mean()),
            "std_defaults": float(defaults.std(ddof=1)) if n_runs > 1 else 0.0,
            "median_defaults": float(np.median(defaults)),
            "max_defaults": float(defaults.max()),
            "frac_full_cascade": float((defaults >= 0.5 * config.N_BANKS).mean()),
            "elapsed_sec": elapsed,
        })
        print(f"rho={rho}: mean_defaults={rows[-1]['mean_defaults']:.2f} "
              f"(n_runs={n_runs}, {elapsed:.1f}s)")
    return pd.DataFrame(rows)


def plot_results(table: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        [str(r) for r in table["rho"]],
        table["mean_defaults"],
        yerr=table["std_defaults"],
        capsize=4,
        color="steelblue",
    )
    ax.set_xlabel("Asset-return correlation (rho)")
    ax.set_ylabel("Average number of bank defaults")
    ax.set_title("Experiment 2: correlation and nonlinear systemic risk\n"
                  f"({int(table['n_runs'].iloc[0])} Monte Carlo runs per configuration)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true",
                         help="Use the paper's full Monte Carlo repetition count "
                              f"(N_RUNS_PAPER={config.N_RUNS_PAPER}) instead of the "
                              f"reduced N_RUNS_RUN={config.N_RUNS_RUN}.")
    args = parser.parse_args()
    n_runs = config.N_RUNS_PAPER if args.full else config.N_RUNS_RUN

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table = run_experiment(n_runs)
    csv_path = RESULTS_DIR / "exp2_correlation_defaults.csv"
    table.to_csv(csv_path, index=False)
    plot_results(table, RESULTS_DIR / "exp2_defaults_by_correlation.png")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
