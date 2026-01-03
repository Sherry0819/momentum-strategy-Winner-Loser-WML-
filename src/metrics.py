from __future__ import annotations

import numpy as np
import pandas as pd


def annualize_return(r: pd.Series, periods_per_year: int = 12) -> float:
    return float(r.mean() * periods_per_year)


def annualize_vol(r: pd.Series, periods_per_year: int = 12) -> float:
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


def sharpe_ratio(r: pd.Series, rf: float = 0.0, periods_per_year: int = 12) -> float:
    """Sharpe ratio using arithmetic mean excess returns."""
    ex = r - rf
    vol = ex.std(ddof=1)
    if vol == 0 or np.isnan(vol):
        return np.nan
    return float((ex.mean() / vol) * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    dd = equity_curve / running_max - 1.0
    return float(dd.min())


def rolling_mean_annualized(r: pd.Series, window_months: int) -> pd.Series:
    return r.rolling(window_months).mean() * 12.0
