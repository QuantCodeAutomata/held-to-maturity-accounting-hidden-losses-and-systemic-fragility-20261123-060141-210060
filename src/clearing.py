"""Interbank payment clearing (Eisenberg & Noe, 2001; Rogers & Veraart, 2013).

Custom implementation of the Greatest Clearing Vector Algorithm (paper,
Sec. "Market Clearing", p. 25): the fictitious-default fixed-point iteration
that starts from the vector of full contractual obligations and decreases
monotonically to the greatest clearing vector, with bankruptcy-cost recovery
rates applied separately to a defaulting bank's external assets (kappa_e)
and to the interbank claims it has received (kappa_l), as in Rogers &
Veraart (2013). Vectorized over an explicit "runs" batch dimension so many
independent Monte Carlo simulations clear simultaneously.
"""

from __future__ import annotations

import numpy as np


def greatest_clearing_vector(
    p_bar: np.ndarray,
    relative_liabilities: np.ndarray,
    external_assets: np.ndarray,
    kappa_l: float,
    kappa_e: float,
    tol: float = 1e-10,
    max_iter: int = 200,
) -> np.ndarray:
    """Solve for the greatest clearing vector p* (Rogers & Veraart, 2013).

    Parameters
    ----------
    p_bar : array (..., n)
        Total contractual interbank obligations of each bank this episode.
    relative_liabilities : array (..., n, n)
        Pi[..., i, j] = L[..., i, j] / p_bar[..., i], the share of bank i's
        total obligation owed to bank j (0 if p_bar_i == 0).
    external_assets : array (..., n)
        Net worth e_i from outside the interbank network (portfolio plus
        capital reserve minus deposits).
    kappa_l, kappa_e : float
        Recovery rates applied, for a defaulting bank, to interbank claims
        received (kappa_l) and to external assets (kappa_e).

    Returns
    -------
    p_star : array (..., n)
        The clearing payment vector: p_star_i = p_bar_i for solvent banks,
        and the haircut available resources for insolvent ones.
    """
    if not (0.0 <= kappa_l <= 1.0 and 0.0 <= kappa_e <= 1.0):
        raise ValueError("recovery rates must be in [0, 1]")
    p = p_bar.copy()
    pi_t = np.swapaxes(relative_liabilities, -1, -2)  # Pi^T[..., i, j] = Pi[..., j, i]
    for _ in range(max_iter):
        incoming = np.einsum("...ij,...j->...i", pi_t, p)
        resources = external_assets + incoming
        solvent = resources >= p_bar - 1e-12
        haircut = kappa_e * np.maximum(external_assets, 0.0) + kappa_l * np.maximum(incoming, 0.0)
        new_p = np.where(solvent, p_bar, haircut)
        new_p = np.clip(new_p, 0.0, p_bar)
        if np.max(np.abs(new_p - p)) < tol:
            p = new_p
            break
        p = new_p
    return p


def equity_after_clearing(
    p_bar: np.ndarray,
    relative_liabilities: np.ndarray,
    external_assets: np.ndarray,
    p_star: np.ndarray,
) -> np.ndarray:
    """Accounting equity after clearing: e_i + incoming_i - p_bar_i.

    This uses the *contractual* obligation p_bar_i (not the haircut payment
    p_star_i), so it is negative exactly for banks identified as defaulting
    (paper, p. 25: "Banks with negative equity after clearing are identified
    as defaulting").
    """
    pi_t = np.swapaxes(relative_liabilities, -1, -2)
    incoming = np.einsum("...ij,...j->...i", pi_t, p_star)
    return external_assets + incoming - p_bar


def cash_received(relative_liabilities: np.ndarray, p_star: np.ndarray) -> np.ndarray:
    """Actual interbank cash received by each bank, given the clearing vector."""
    pi_t = np.swapaxes(relative_liabilities, -1, -2)
    return np.einsum("...ij,...j->...i", pi_t, p_star)
