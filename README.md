# Asset Pricing Mini‑Suite: Efficient Frontier + Industry Momentum + Accrual Factor

## What’s inside

### Part I — Mean‑Variance Efficient Frontier (12 Industries)
- Load monthly returns on 12 Ken French industry portfolios
- Compute and plot:
  - **Unconstrained** efficient frontier
  - **Long‑only** efficient frontier
- Estimate **tangency portfolio** weights (max Sharpe)

**Output**
- `reports/figures/efficient_frontier.png`
- `reports/tangency_weights.csv`

### Part II — Industry Momentum (Winner–Loser / WML)
- Monthly rebalancing starting 1950‑01
- Momentum signal: average return over **t‑12 to t‑2** (skip the most recent month)
- Winner: highest signal industry, Loser: lowest signal industry
- WML = Winner − Loser
- Rolling mean plots and Winner/Loser frequency + turnover

**Output**
- `reports/momentum_summary.csv`
- `reports/figures/momentum_wml_rolling.png`
- `reports/figures/momentum_winner_frequency.png`
- `reports/figures/momentum_loser_frequency.png`
- `reports/momentum_turnover.csv`

### Part III — Accrual Factor (optional; requires proprietary WRDS exports)
- Build a value‑weighted accrual factor each month:
  - Top 10% accrual = “High”
  - Bottom 10% accrual = “Low”
  - Factor = Low − High
- Report mean/std and a one‑sample t‑test

**Output**
- `reports/accrual_factor.csv`
- `reports/accrual_factor_stats.csv`
- `reports/figures/accrual_factor_wealth.png`

---

## Quickstart

### 1) Create a virtual environment + install
```bash
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

### 2) Add data
See `data/README.md`.

### 3) Run
```bash
python run.py
```

To include Part III (if you have the CSVs):
```bash
python run.py --run_accrual
```
