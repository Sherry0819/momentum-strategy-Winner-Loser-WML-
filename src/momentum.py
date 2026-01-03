from __future__ import annotations

import pandas as pd
from dataclasses import dataclass


@dataclass
class MomentumResult:
    winner_ret: pd.Series
    loser_ret: pd.Series
    wml_ret: pd.Series
    winner_industry: pd.Series
    loser_industry: pd.Series
    signal: pd.DataFrame


def industry_momentum_wml(
    rets_pct: pd.DataFrame,
    start_date: str = "1950-01-01",
    lookback_months: int = 12,
    skip_months: int = 1,
) -> MomentumResult:
    """
    Momentum signal: rolling mean over lookback window, shifted by skip_months.
    Winner = highest signal; Loser = lowest signal.
    Returns are in percent per month (Ken French style).
    """
    rets_pct = rets_pct.copy()
    rets_pct.index = pd.DatetimeIndex(rets_pct.index)
    rets_pct = rets_pct.sort_index()

    signal = rets_pct.rolling(window=lookback_months).mean().shift(skip_months)

    signal = signal.loc[pd.to_datetime(start_date):]
    rets_pct = rets_pct.loc[signal.index.min():]

    winner_idx = signal.idxmax(axis=1)
    loser_idx = signal.idxmin(axis=1)

    winner_ret = pd.Series([rets_pct.loc[t, winner_idx.loc[t]] for t in signal.index], index=signal.index, name="Winner")
    loser_ret = pd.Series([rets_pct.loc[t, loser_idx.loc[t]] for t in signal.index], index=signal.index, name="Loser")
    wml = (winner_ret - loser_ret).rename("WML")

    return MomentumResult(
        winner_ret=winner_ret,
        loser_ret=loser_ret,
        wml_ret=wml,
        winner_industry=winner_idx.rename("WinnerIndustry"),
        loser_industry=loser_idx.rename("LoserIndustry"),
        signal=signal,
    )


def turnover_from_picks(picks: pd.Series) -> float:
    return float((picks != picks.shift(1)).mean())
