from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class AccrualFactorResult:
    factor: pd.Series
    high_ret: pd.Series
    low_ret: pd.Series
    mean: float
    std: float
    t_stat: float
    p_value: float


def build_accrual_factor(crsp_like_csv: str | Path, accrual_csv: str | Path, top_bottom: float = 0.10) -> AccrualFactorResult:
    """
    Build a value-weighted accrual factor each month.

    Expected columns:
      - crsp_like_csv: date, PERMNO, RET, PRC, SHROUT
      - accrual_csv:   permno, public_date, accrual

    Factor definition:
      LowAccrual - HighAccrual, where portfolios are the bottom/top `top_bottom` fraction.
    """
    crsp_like_csv = Path(crsp_like_csv)
    accrual_csv = Path(accrual_csv)
    if not crsp_like_csv.exists() or not accrual_csv.exists():
        raise FileNotFoundError(
            "Missing Part III input files. Put them in data/raw/ as: data/raw/AS2P3_1.csv and data/raw/accrual_rate.csv"
        )

    df1 = pd.read_csv(crsp_like_csv)
    df2 = pd.read_csv(accrual_csv)

    df1["date"] = pd.to_datetime(df1["date"])
    df2["public_date"] = pd.to_datetime(df2["public_date"])

    for col in ["RET", "PRC", "SHROUT"]:
        df1[col] = pd.to_numeric(df1[col], errors="coerce")
    df2["accrual"] = pd.to_numeric(df2["accrual"], errors="coerce")

    df1["MC"] = df1["PRC"].abs() * df1["SHROUT"]
    df1["ym"] = df1["date"].dt.to_period("M")
    df2["ym"] = df2["public_date"].dt.to_period("M")

    df = pd.merge(
        df1[["date", "ym", "PERMNO", "RET", "MC"]],
        df2[["ym", "permno", "accrual"]],
        left_on=["ym", "PERMNO"],
        right_on=["ym", "permno"],
        how="inner",
    ).dropna(subset=["RET", "MC", "accrual"])

    idx, high_ret, low_ret, factor = [], [], [], []

    for ym, g in df.groupby("ym"):
        if len(g) < 10:
            continue
        n = len(g)
        k = max(1, int(np.floor(n * top_bottom)))

        high = g.nlargest(k, "accrual")
        low = g.nsmallest(k, "accrual")

        def vw_return(x: pd.DataFrame) -> float:
            w = x["MC"] / x["MC"].sum()
            return float((w * x["RET"]).sum())

        r_high = vw_return(high)
        r_low = vw_return(low)

        idx.append(ym.to_timestamp("M"))
        high_ret.append(r_high)
        low_ret.append(r_low)
        factor.append(r_low - r_high)

    high_s = pd.Series(high_ret, index=pd.DatetimeIndex(idx), name="HighAccrual")
    low_s = pd.Series(low_ret, index=pd.DatetimeIndex(idx), name="LowAccrual")
    fac_s = pd.Series(factor, index=pd.DatetimeIndex(idx), name="AccrualFactor")

    mean = float(fac_s.mean())
    std = float(fac_s.std(ddof=1))
    t_stat, p_value = stats.ttest_1samp(fac_s.dropna().values, 0.0)

    return AccrualFactorResult(
        factor=fac_s,
        high_ret=high_s,
        low_ret=low_s,
        mean=mean,
        std=std,
        t_stat=float(t_stat),
        p_value=float(p_value),
    )
