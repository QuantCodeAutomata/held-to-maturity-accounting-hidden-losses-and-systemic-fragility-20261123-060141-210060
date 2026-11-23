"""Experiment 3: timing of loss recognition, HTM (concentrated) vs. MTM (gradual).

Runs the agent-based interbank model (`src.abm.run_simulation`) at a fixed
asset-return correlation (`src.config.EXP3_RHO`, p. 29) while spreading the
same total shock magnitude over 1, 2 or 3 consecutive shock episodes ending
at the pre-shock horizon (`src.config.EXP3_SHOCK_TIMINGS`): a single shock
mimics concentrated Held-To-Maturity loss recognition, several smaller
shocks mimic gradual Mark-To-Market recognition of the same total loss.

Usage
-----
    python -m exp.exp_3_timing_of_loss_recognition [--full]

Without ``--full`` the executed number of Monte Carlo repetitions is
``src.config.N_RUNS_RUN``; with ``--full`` it is the paper's
``src.config.N_RUNS_PAPER`` (2000 runs per configuration, p. 29).
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
    """Run the model at each shock-splitting count in `config.EXP3_SHOCK_TIMINGS`.

    Returns one row per number of shock episodes, with summary statistics
    of the number of defaults over `n_runs` Monte Carlo repetitions. The
    total shock variance is held constant across rows (see
    `src.abm.run_simulation`'s 1/sqrt(k) scaling, paper's "same total shock
    magnitude").
    """
    rows = []
    for n_shocks in config.EXP3_SHOCK_TIMINGS:
        shock_episodes = tuple(range(1, n_shocks + 1))
        start = time.time()
        result = run_simulation(
            n_runs=n_runs,
            rho=config.EXP3_RHO,
            shock_episodes=shock_episodes,
            n_banks=config.N_BANKS,
            pre_shock_episodes=config.PRE_SHOCK_EPISODES,
            delta_sigma=config.SHOCK_DELTA_SIGMA,
            seed=config.RANDOM_SEED,
        )
        elapsed = time.time() - start
        defaults = result.n_defaults
        rows.append({
            "n_shock_episodes": n_shocks,
            "regime": "HTM (concentrated)" if n_shocks == 1 else "MTM (gradual)",
            "n_runs": n_runs,
            "n_banks": config.N_BANKS,
            "mean_defaults": float(defaults.mean()),
            "std_defaults": float(defaults.std(ddof=1)) if n_runs > 1 else 0.0,
            "median_defaults": float(np.median(defaults)),
            "max_defaults": float(defaults.max()),
            "elapsed_sec": elapsed,
        })
        print(f"n_shock_episodes={n_shocks}: mean_defaults={rows[-1]['mean_defaults']:.2f} "
              f"(n_runs={n_runs}, {elapsed:.1f}s)")
    return pd.DataFrame(rows)


def plot_results(table: pd.DataFrame, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(
        [str(k) for k in table["n_shock_episodes"]],
        table["mean_defaults"],
        yerr=table["std_defaults"],
        capsize=4,
        color="indianred",
    )
    ax.set_xlabel("Number of episodes the total shock is split across")
    ax.set_ylabel("Average number of bank defaults")
    ax.set_title("Experiment 3: timing of loss recognition (HTM vs. MTM)\n"
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
    csv_path = RESULTS_DIR / "exp3_shock_timing_defaults.csv"
    table.to_csv(csv_path, index=False)
    plot_results(table, RESULTS_DIR / "exp3_defaults_by_shock_timing.png")
    print(f"Saved {csv_path}")


if __name__ == "__main__":
    main()
