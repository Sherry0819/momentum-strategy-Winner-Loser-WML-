from __future__ import annotations

import pandas as pd
import statsmodels.api as sm
from dataclasses import dataclass


@dataclass
class FFAlphaResult:
    alpha: float
    alpha_t: float
    params: pd.Series
    tstats: pd.Series


def ff3_alpha(port_ret_pct: pd.Series, ff_factors_pct: pd.DataFrame) -> FFAlphaResult:
    """FF3 regression on monthly percent data: (R - RF) ~ Mkt-RF + SMB + HML."""
    df = pd.concat([port_ret_pct, ff_factors_pct], axis=1, join="inner").dropna()
    y = df[port_ret_pct.name] - df["RF"]
    X = sm.add_constant(df[["Mkt-RF", "SMB", "HML"]])
    model = sm.OLS(y, X).fit()

    return FFAlphaResult(
        alpha=float(model.params["const"]),
        alpha_t=float(model.tvalues["const"]),
        params=model.params,
        tstats=model.tvalues,
    )
