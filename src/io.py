from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pandas as pd


@dataclass
class IndustryReturns:
    dates: pd.DatetimeIndex
    returns_pct: pd.DataFrame  # percent per month, 12 industries
    returns: pd.DataFrame      # decimal per month


def load_industry_returns_xlsx(path: str | Path) -> IndustryReturns:
    """
    Load the 12-industry monthly returns spreadsheet used in class.

    Expected layout (common in Ken French exports):
      - Column A: dates as YYYYMM
      - Columns B-M: 12 industry returns in percent per month

    Returns:
      IndustryReturns with both percent and decimal returns.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}. Put it at data/raw/industry_returns.xlsx")

    dates = pd.read_excel(path, usecols="A", header=1)
    dates.columns = ["date"]
    dates["date"] = pd.to_datetime(dates["date"].astype(str), format="%Y%m")

    rets_pct = pd.read_excel(path, usecols="B:M", header=1)
    rets_pct = rets_pct.apply(pd.to_numeric, errors="coerce")
    rets_pct.index = pd.DatetimeIndex(dates["date"])
    rets_pct = rets_pct.dropna(how="all")

    rets = rets_pct / 100.0
    return IndustryReturns(dates=rets.index, returns_pct=rets_pct, returns=rets)
