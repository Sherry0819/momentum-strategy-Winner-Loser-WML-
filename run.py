from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.io import load_industry_returns_xlsx
from src.efficient_frontier import efficient_frontier, tangency_portfolio
from src.momentum import industry_momentum_wml, turnover_from_picks
from src.metrics import rolling_mean_annualized
from src.accrual_factor import build_accrual_factor


def main():
    parser = argparse.ArgumentParser(description="Asset Pricing mini-suite (Efficient Frontier, Momentum, Accrual Factor).")
    parser.add_argument("--industry_xlsx", type=str, default="data/raw/industry_returns.xlsx")
    parser.add_argument("--rf_annual", type=float, default=0.04, help="Annual risk-free rate (decimal), default 4%%.")
    parser.add_argument("--outdir", type=str, default="reports/figures")
    parser.add_argument("--run_accrual", action="store_true", help="Run Part III if proprietary CSVs are available.")
    parser.add_argument("--crsp_csv", type=str, default="data/raw/AS2P3_1.csv")
    parser.add_argument("--accrual_csv", type=str, default="data/raw/accrual_rate.csv")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    Path("reports").mkdir(exist_ok=True)

    # -----------------------
    # Part I: Efficient Frontier
    # -----------------------
    ind = load_industry_returns_xlsx(args.industry_xlsx)
    ret = ind.returns  # decimals
    mu = ret.mean()    # monthly mean
    Sigma = ret.cov()  # monthly covariance

    rf_monthly = args.rf_annual / 12.0

    mu_targets = np.arange(0.0, 0.0151, 0.001)  # 0% to 1.5% per month
    fr_uncon = efficient_frontier(mu, Sigma, mu_targets, long_only=False)
    fr_long = efficient_frontier(mu, Sigma, mu_targets, long_only=True)

    w_tan_uncon = tangency_portfolio(mu, Sigma, rf_monthly, long_only=False)
    w_tan_long = tangency_portfolio(mu, Sigma, rf_monthly, long_only=True)

    plt.figure()
    plt.plot(fr_uncon.sigma_targets, fr_uncon.mu_targets, label="Unconstrained")
    plt.plot(fr_long.sigma_targets, fr_long.mu_targets, label="Long-only")
    plt.xlabel("Monthly Volatility (sigma)")
    plt.ylabel("Monthly Expected Return (mu)")
    plt.title("Mean-Variance Efficient Frontier (12 Industry Portfolios)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "efficient_frontier.png", dpi=200)
    plt.close()

    weights = pd.DataFrame(
        {"unconstrained": w_tan_uncon, "long_only": w_tan_long},
        index=ret.columns,
    )
    weights.to_csv("reports/tangency_weights.csv")

    # -----------------------
    # Part II: Momentum (WML)
    # -----------------------
    mom = industry_momentum_wml(ind.returns_pct, start_date="1950-01-01", lookback_months=12, skip_months=1)

    mom_stats = pd.DataFrame({
        "mean_ann_%": 12 * pd.concat([mom.winner_ret, mom.loser_ret, mom.wml_ret]).mean(),
        "vol_ann_%": (np.sqrt(12) * pd.concat([mom.winner_ret, mom.loser_ret, mom.wml_ret]).std(ddof=1)),
    })
    mom_stats.to_csv("reports/momentum_summary.csv")

    roll3 = rolling_mean_annualized(mom.wml_ret, 36)
    roll10 = rolling_mean_annualized(mom.wml_ret, 120)

    plt.figure()
    plt.plot(roll3.index, roll3.values, label="Rolling 3Y mean (ann., %)")
    plt.plot(roll10.index, roll10.values, label="Rolling 10Y mean (ann., %)")
    plt.axhline(0, linewidth=0.8)
    plt.title("Industry Momentum WML: Rolling Average Returns")
    plt.xlabel("Date")
    plt.ylabel("Annualized return (%)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "momentum_wml_rolling.png", dpi=200)
    plt.close()

    win_counts = mom.winner_industry.value_counts().sort_values(ascending=False)
    los_counts = mom.loser_industry.value_counts().sort_values(ascending=False)

    plt.figure()
    win_counts.plot(kind="bar")
    plt.title("Winner Industry Frequency")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outdir / "momentum_winner_frequency.png", dpi=200)
    plt.close()

    plt.figure()
    los_counts.plot(kind="bar")
    plt.title("Loser Industry Frequency")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outdir / "momentum_loser_frequency.png", dpi=200)
    plt.close()

    turnover_winner = turnover_from_picks(mom.winner_industry)
    turnover_loser = turnover_from_picks(mom.loser_industry)
    pd.Series({"winner_turnover": turnover_winner, "loser_turnover": turnover_loser}).to_csv("reports/momentum_turnover.csv")

    # -----------------------
    # Part III: Accrual factor (optional)
    # -----------------------
    if args.run_accrual:
        res = build_accrual_factor(args.crsp_csv, args.accrual_csv, top_bottom=0.10)
        res.factor.to_csv("reports/accrual_factor.csv")
        pd.Series({"mean": res.mean, "std": res.std, "t_stat": res.t_stat, "p_value": res.p_value}).to_csv("reports/accrual_factor_stats.csv")

        plt.figure()
        equity = (1 + res.factor.fillna(0) / 100.0).cumprod()
        plt.plot(equity.index, equity.values)
        plt.title("Accrual Factor: Wealth Index (Low - High)")
        plt.xlabel("Date")
        plt.ylabel("Wealth")
        plt.tight_layout()
        plt.savefig(outdir / "accrual_factor_wealth.png", dpi=200)
        plt.close()

    print("Done. Outputs saved under reports/ and reports/figures/.")


if __name__ == "__main__":
    main()
