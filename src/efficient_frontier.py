from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.optimize import minimize


@dataclass
class FrontierResult:
    mu_targets: np.ndarray
    sigma_targets: np.ndarray
    weights: np.ndarray  # (len(mu_targets), n)


def _portfolio_mean(w: np.ndarray, mu: np.ndarray) -> float:
    return float(w @ mu)


def _portfolio_var(w: np.ndarray, Sigma: np.ndarray) -> float:
    return float(w @ Sigma @ w)


def tangency_portfolio(mu: pd.Series, Sigma: pd.DataFrame, rf: float, long_only: bool = False) -> np.ndarray:
    """Max Sharpe portfolio with sum(w)=1; optional long-only bounds."""
    mu_vec = mu.values
    S = Sigma.values
    n = len(mu_vec)

    def neg_sharpe(w: np.ndarray) -> float:
        m = _portfolio_mean(w, mu_vec) - rf
        v = _portfolio_var(w, S)
        if v <= 0:
            return 1e9
        return -(m / np.sqrt(v))

    w0 = np.ones(n) / n
    cons = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    bounds = [(0.0, 1.0)] * n if long_only else None

    res = minimize(neg_sharpe, w0, constraints=cons, bounds=bounds, method="SLSQP")
    if not res.success:
        raise RuntimeError(f"Tangency optimization failed: {res.message}")
    return res.x


def efficient_frontier(mu: pd.Series, Sigma: pd.DataFrame, mu_targets: np.ndarray, long_only: bool = False) -> FrontierResult:
    """Min-variance frontier for a grid of target means."""
    mu_vec = mu.values
    S = Sigma.values
    n = len(mu_vec)

    weights = np.zeros((len(mu_targets), n))
    sigmas = np.zeros(len(mu_targets))

    for k, target in enumerate(mu_targets):
        w0 = np.ones(n) / n
        cons = [
            {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
            {"type": "eq", "fun": lambda w, t=target: (w @ mu_vec) - t},
        ]
        bounds = [(0.0, 1.0)] * n if long_only else None

        res = minimize(lambda w: _portfolio_var(w, S), w0, constraints=cons, bounds=bounds, method="SLSQP")
        if not res.success:
            weights[k, :] = np.nan
            sigmas[k] = np.nan
            continue

        w_opt = res.x
        weights[k, :] = w_opt
        sigmas[k] = np.sqrt(_portfolio_var(w_opt, S))

    return FrontierResult(mu_targets=mu_targets, sigma_targets=sigmas, weights=weights)
