"""Agent-based interbank model shared by Experiments 2 and 3.

Implements the dynamic interbank system described on pp. 18-29 of the
paper: endogenous network formation through Dirichlet-categorical
counterparty selection with belief updating (Eq. 5-6), correlated risky
portfolio returns (Eq. 7), a systemic shock (Eq. 8), and clearing through
the Rogers & Veraart (2013) Greatest Clearing Vector Algorithm (`src.clearing`).

The simulation is vectorized over an explicit "runs" axis (independent
Monte Carlo repetitions of the same configuration run side by side as numpy
arrays) rather than built on an agent-stepping framework such as `mesa`,
because the paper's headline statistic -- the average number of defaults
over thousands of independent runs of a 100-bank, 300-episode network -- is
a Monte Carlo average that this project's compute budget can only reach
with vectorized array operations; each bank is still carried as an
individual state vector entry with its own balance sheet and belief matrix,
matching the paper's agent-based description.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from src import config
from src.clearing import equity_after_clearing, greatest_clearing_vector


@dataclass
class SimulationResult:
    """Outcome of one batch of Monte Carlo runs of the interbank model."""

    n_defaults: np.ndarray  # (n_runs,) final number of defaulted banks per run
    equity_path: np.ndarray  # (n_episodes_recorded, n_runs) mean equity of survivors
    defaulted_mask_final: np.ndarray  # (n_runs, n_banks) bool, who defaulted


def _initial_balance_sheets(rng: np.random.Generator, n_runs: int, n_banks: int) -> dict:
    """Heterogeneous starting deposits, capital reserves and equity.

    The paper calibrates the model "to match empirical moments of asset
    distribution, leverage, and network topology" (Sec. "Calibration") but
    does not print the deposit distribution or the initial leverage used to
    seed the system; a lognormal deposit size (a standard heavy-tailed proxy
    for the skewed distribution of bank balance-sheet size) and an initial
    leverage of half the regulatory maximum are used as documented
    assumptions ("not specified in the paper").
    """
    deposits = rng.lognormal(mean=0.0, sigma=1.0, size=(n_runs, n_banks)) * 100.0
    initial_leverage = config.LEVERAGE_MAX / 2.0  # not specified in the paper
    equity = deposits / (initial_leverage - 1.0)
    capital_reserve = config.CAPITAL_RESERVE_RATIO * deposits
    portfolio = equity + deposits - capital_reserve  # Assets = Portfolio + reserve, Liab = Deposits
    return {
        "deposits": deposits,
        "capital_reserve": capital_reserve,
        "portfolio": portfolio,
    }


def _correlated_normal_draws(rng: np.random.Generator, n_runs: int, n_banks: int, rho: float) -> np.ndarray:
    """Z ~ N(0, Sigma) with a constant off-diagonal correlation rho (paper Eq. 7).

    Returns an array of shape (n_runs, n_banks) of standard-normal variates
    correlated across banks through a single common factor, equivalent to
    the Cholesky factorization Sigma = L L^T of a constant-correlation
    matrix (used instead of forming the full n_banks x n_banks matrix, which
    is algebraically identical for a one-factor/constant-correlation
    structure and far cheaper for n_banks=100).
    """
    if not (0.0 <= rho <= 1.0):
        raise ValueError("rho must lie in [0, 1]")
    common = rng.standard_normal((n_runs, 1))
    idiosyncratic = rng.standard_normal((n_runs, n_banks))
    return np.sqrt(rho) * common + np.sqrt(1.0 - rho) * idiosyncratic


def _dirichlet_batch(rng: np.random.Generator, alpha: np.ndarray) -> np.ndarray:
    """Batched Dirichlet sampling: one draw per (run, bank) row of `alpha`.

    numpy has no batched `dirichlet`, so each Dirichlet(alpha) draw is built
    from independent Gamma(alpha_k, 1) draws normalized to sum to one (the
    standard Gamma-ratio construction of a Dirichlet random vector), which
    numpy's `gamma` supports with a fully broadcast shape parameter.
    """
    gamma_draws = rng.gamma(shape=np.maximum(alpha, 1e-6))
    total = gamma_draws.sum(axis=-1, keepdims=True)
    # After many episodes of memory decay (Eq. 6) a row's concentrations can
    # all underflow below double precision, so every Gamma(shape) draw in
    # that row rounds to exactly 0.0; fall back to a uniform row instead of
    # propagating 0/0 (NaN) into counterparty selection.
    safe_total = np.where(total > 0, total, 1.0)
    probs = gamma_draws / safe_total
    uniform = np.full_like(probs, 1.0 / probs.shape[-1])
    return np.where(total > 0, probs, uniform)


def _sample_counterparty_counts(rng: np.random.Generator, probs: np.ndarray, n_requests: int) -> np.ndarray:
    """For each (run, bank) row of `probs`, draw `n_requests` i.i.d. counterparties.

    Returns counts of shape (..., n_banks): how many of the n_requests
    borrowing requests of bank i (in a given run) were routed to each
    possible counterparty j. Counting n_requests i.i.d. categorical draws
    from `probs` is exactly a Multinomial(n_requests, probs) draw, so
    `Generator.multinomial` (batched over the leading (run, bank) axes) is
    used instead of materializing n_requests one-hot draws per row, which
    is algebraically identical and orders of magnitude cheaper for
    n_runs x n_banks x n_banks arrays.
    """
    probs = probs / probs.sum(axis=-1, keepdims=True)  # guard against floating-point shortfall
    return rng.multinomial(n_requests, probs)


def _build_relative_liabilities(loan_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """p_bar (total owed per borrower) and Pi (relative-liability matrix) from `loan_matrix`."""
    p_bar = loan_matrix.sum(axis=-1)
    safe_p_bar = np.where(p_bar > 0, p_bar, 1.0)
    pi = loan_matrix / safe_p_bar[..., None]
    pi = np.where(p_bar[..., None] > 0, pi, 0.0)
    return p_bar, pi


def run_simulation(
    n_runs: int,
    rho: float,
    shock_episodes: tuple[int, ...],
    *,
    n_banks: int = config.N_BANKS,
    pre_shock_episodes: int = config.PRE_SHOCK_EPISODES,
    delta_sigma: float = config.SHOCK_DELTA_SIGMA,
    seed: int = config.RANDOM_SEED,
    record_every: int = 50,
) -> SimulationResult:
    """Run `n_runs` independent copies of the interbank model to the given shock design.

    Parameters
    ----------
    n_runs : number of independent Monte Carlo simulations (paper p. 27/29: 2000).
    rho : constant pairwise correlation of banks' risky-asset returns (Eq. 7).
    shock_episodes : how many of the final pre-shock episodes carry the adverse
        shock instead of a normal return draw, e.g. (1,) for a single shock at
        t=pre_shock_episodes, or (1, 2) for two consecutive shock episodes
        ending at t=pre_shock_episodes (paper, Experiment 3: HTM vs MTM
        accounting timing).
    record_every : how often (in episodes) to record the mean survivor equity,
        used only for diagnostic plots.

    The total shock magnitude is held constant across different numbers of
    shock episodes (paper, Experiment 3: "the same total shock magnitude" is
    split across 1, 2 or 3 episodes). Each episode's loss is one-sided,
    R = -|Delta_sigma * sigma * Z| (Eq. 8), so the k per-episode draws add up
    rather than partially cancel: for k independent episodes each scaled by a
    factor c, the expected total loss is k * c * sigma * E|Z|. Keeping that
    sum equal to the single-episode magnitude delta_sigma * sigma * E|Z|
    requires c = 1/k (not 1/sqrt(k), which instead holds the *variance* of a
    zero-mean sum constant and is not the right normalization for a sum of
    one-sided absolute shocks).
    """
    rng = np.random.default_rng(seed)
    sheets = _initial_balance_sheets(rng, n_runs, n_banks)
    portfolio = sheets["portfolio"]
    deposits = sheets["deposits"]
    reserve = sheets["capital_reserve"]
    defaulted = np.zeros((n_runs, n_banks), dtype=bool)

    alpha = np.full((n_runs, n_banks, n_banks), config.DIRICHLET_PRIOR_CONCENTRATION)
    eye = np.eye(n_banks, dtype=bool)
    alpha[:, eye] = 0.0  # a bank never lends to itself

    n_shocks = len(shock_episodes)
    shock_delta_sigma = delta_sigma / n_shocks
    last_shock_episode = max(shock_episodes)
    shock_set = {pre_shock_episodes - last_shock_episode + s for s in shock_episodes}

    daily_sigma = config.PORTFOLIO_SIGMA / np.sqrt(config.EPISODES_PER_YEAR)
    equity_path = []
    for episode in range(1, pre_shock_episodes + 1):
        active = ~defaulted

        # Eq. 7/8: the risky Portfolio ("investable component") evolves on its
        # own existing size; interbank borrowing (below) is a liability/claim
        # settled through capital reserve, not a re-invested leveraged stake,
        # so it cannot compound into the risky return every episode.
        z = _correlated_normal_draws(rng, n_runs, n_banks, rho)
        if episode in shock_set:
            # Eq. 8: R = -|Delta_sigma * L Z|, using the same per-episode
            # sigma as the normal-period draws so Delta_sigma is a
            # volatility *spike* relative to one episode's usual risk, not an
            # inconsistent second re-annualization of sigma.
            returns = -np.abs(shock_delta_sigma * daily_sigma * z)
        else:
            returns = config.PORTFOLIO_MU / config.EPISODES_PER_YEAR + daily_sigma * z
        portfolio = np.where(active, portfolio * (1.0 + returns), portfolio)

        equity = portfolio + reserve - deposits
        equity_floor = np.where(active, equity, 1.0)  # avoid borrowing by defaulted banks

        assets = portfolio + reserve
        max_assets = config.LEVERAGE_MAX * np.maximum(equity_floor, 0.0)
        borrow_cap = np.maximum(max_assets - assets, 0.0)
        reserve_shortfall = np.maximum(config.CAPITAL_RESERVE_RATIO * deposits - reserve, 0.0)
        target_borrow = np.minimum(config.BORROW_SIZE_RATIO_OF_EQUITY * reserve_shortfall, borrow_cap)
        target_borrow = np.where(active, target_borrow, 0.0)

        active_pair_mask = active[:, :, None] & active[:, None, :] & ~eye[None, :, :]
        masked_alpha = np.where(active_pair_mask, alpha, 0.0)
        row_sum = masked_alpha.sum(axis=-1, keepdims=True)
        masked_alpha = np.where(row_sum > 0, masked_alpha, 1.0 / max(n_banks - 1, 1))
        probs = _dirichlet_batch(rng, masked_alpha)
        counts = _sample_counterparty_counts(rng, probs, config.N_BORROW_REQUESTS)

        loan_matrix = target_borrow[:, :, None] * counts / config.N_BORROW_REQUESTS
        alpha = config.BELIEF_MEMORY_DECAY_Q * alpha + counts

        # Loan origination: a borrowed amount moves cash from lender to
        # borrower's capital reserve, it is not invested in the risky
        # Portfolio; the obligation (p_bar) is cleared the same episode
        # (Rogers & Veraart 2013 framework), representing an overnight
        # interbank market.
        reserve = reserve + loan_matrix.sum(axis=-1) - loan_matrix.sum(axis=-2)

        p_bar, pi = _build_relative_liabilities(loan_matrix)
        external_assets = portfolio + reserve - deposits
        p_star = greatest_clearing_vector(p_bar, pi, external_assets, config.KAPPA_L, config.KAPPA_E)
        equity_after = equity_after_clearing(p_bar, pi, external_assets, p_star)
        incoming = np.einsum("...ij,...j->...i", np.swapaxes(pi, -1, -2), p_star)

        newly_defaulted = active & (equity_after < 0.0)
        defaulted = defaulted | newly_defaulted

        reserve = reserve + incoming - p_star
        portfolio = np.where(defaulted, 0.0, portfolio)
        deposits = np.where(defaulted, 0.0, deposits)
        reserve = np.where(defaulted, 0.0, reserve)
        alpha[defaulted] = 0.0

        if episode % record_every == 0 or episode == pre_shock_episodes:
            survivor_equity = np.where(~defaulted, portfolio + reserve - deposits, np.nan)
            any_survivor = (~defaulted).any(axis=-1)
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="Mean of empty slice")
                mean_equity = np.nanmean(survivor_equity, axis=-1)
            # A run in which every bank has defaulted has no survivors to
            # average; record NaN explicitly instead of letting nanmean warn
            # on an all-NaN slice.
            equity_path.append(np.where(any_survivor, mean_equity, np.nan))

    n_defaults = defaulted.sum(axis=-1).astype(float)
    return SimulationResult(
        n_defaults=n_defaults,
        equity_path=np.array(equity_path),
        defaulted_mask_final=defaulted,
    )
