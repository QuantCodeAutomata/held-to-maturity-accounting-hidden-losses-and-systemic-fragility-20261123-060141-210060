"""Unit tests for the Rogers & Veraart (2013) Greatest Clearing Vector Algorithm."""

from __future__ import annotations

import numpy as np

from src.clearing import cash_received, equity_after_clearing, greatest_clearing_vector


def test_no_obligations_clears_to_zero():
    p_bar = np.zeros(3)
    pi = np.zeros((3, 3))
    external = np.array([10.0, 5.0, -2.0])
    p_star = greatest_clearing_vector(p_bar, pi, external, kappa_l=0.9, kappa_e=0.5)
    assert np.allclose(p_star, 0.0)


def test_fully_solvent_ring_pays_in_full():
    """A ring A->B->C->A where every bank has enough external assets to cover its debt."""
    n = 3
    p_bar = np.array([1.0, 1.0, 1.0])
    pi = np.zeros((n, n))
    pi[0, 1] = 1.0
    pi[1, 2] = 1.0
    pi[2, 0] = 1.0
    external = np.array([5.0, 5.0, 5.0])
    p_star = greatest_clearing_vector(p_bar, pi, external, kappa_l=0.9, kappa_e=0.5)
    assert np.allclose(p_star, p_bar, atol=1e-6)
    equity = equity_after_clearing(p_bar, pi, external, p_star)
    assert np.all(equity > 0)


def test_insolvent_bank_pays_less_than_owed_and_has_negative_equity():
    n = 2
    p_bar = np.array([10.0, 0.0])
    pi = np.zeros((n, n))
    pi[0, 1] = 1.0  # bank 0 owes its whole p_bar to bank 1
    external = np.array([-20.0, 0.0])  # bank 0 has no means to pay
    p_star = greatest_clearing_vector(p_bar, pi, external, kappa_l=0.9, kappa_e=0.5)
    assert p_star[0] < p_bar[0]
    equity = equity_after_clearing(p_bar, pi, external, p_star)
    assert equity[0] < 0.0


def test_clearing_vector_is_monotone_in_external_assets():
    """Greater external assets can only weakly increase the clearing vector (RV2013 Prop. 1)."""
    n = 4
    rng = np.random.default_rng(0)
    p_bar = rng.uniform(1.0, 5.0, size=n)
    pi = rng.uniform(0.0, 1.0, size=(n, n))
    np.fill_diagonal(pi, 0.0)
    pi = pi / np.maximum(pi.sum(axis=-1, keepdims=True), 1e-12)
    low_external = rng.uniform(-3.0, 0.0, size=n)
    high_external = low_external + 2.0
    p_low = greatest_clearing_vector(p_bar, pi, low_external, kappa_l=0.9, kappa_e=0.5)
    p_high = greatest_clearing_vector(p_bar, pi, high_external, kappa_l=0.9, kappa_e=0.5)
    assert np.all(p_high >= p_low - 1e-8)


def test_batched_runs_axis_matches_independent_calls():
    n, n_runs = 3, 5
    rng = np.random.default_rng(1)
    p_bar = rng.uniform(1.0, 5.0, size=(n_runs, n))
    pi = rng.uniform(0.0, 1.0, size=(n_runs, n, n))
    for r in range(n_runs):
        np.fill_diagonal(pi[r], 0.0)
    pi = pi / np.maximum(pi.sum(axis=-1, keepdims=True), 1e-12)
    external = rng.uniform(-2.0, 2.0, size=(n_runs, n))

    batched = greatest_clearing_vector(p_bar, pi, external, kappa_l=0.9, kappa_e=0.5)
    for r in range(n_runs):
        single = greatest_clearing_vector(p_bar[r], pi[r], external[r], kappa_l=0.9, kappa_e=0.5)
        assert np.allclose(batched[r], single, atol=1e-8)


def test_cash_received_nonnegative_and_bounded_by_obligations():
    n = 3
    rng = np.random.default_rng(2)
    p_bar = rng.uniform(1.0, 5.0, size=n)
    pi = rng.uniform(0.0, 1.0, size=(n, n))
    np.fill_diagonal(pi, 0.0)
    pi = pi / np.maximum(pi.sum(axis=-1, keepdims=True), 1e-12)
    external = rng.uniform(-2.0, 2.0, size=n)
    p_star = greatest_clearing_vector(p_bar, pi, external, kappa_l=0.9, kappa_e=0.5)
    received = cash_received(pi, p_star)
    assert np.all(received >= -1e-8)
    assert np.all(received <= p_bar.sum() + 1e-6)


def test_invalid_recovery_rate_raises():
    p_bar = np.zeros(2)
    pi = np.zeros((2, 2))
    external = np.zeros(2)
    try:
        greatest_clearing_vector(p_bar, pi, external, kappa_l=1.5, kappa_e=0.5)
        assert False, "expected ValueError"
    except ValueError:
        pass
