import os
import numpy as np
import pandas as pd
import cvxpy as cp
from sqlalchemy import create_engine, text
import ecos 

# ---------- CONFIG ----------
ASSET_TYPES = ("EQUITY", "INDEX", "MUTUALFUND")
MIN_HISTORY_DAYS = 252
START_DATE = None
END_DATE   = None
CONF_LEVEL = 0.95


def compute_portfolio_daily_returns(
    asset_types=ASSET_TYPES,
    min_history_days=MIN_HISTORY_DAYS,
    start_date=START_DATE,
    end_date=END_DATE,
    conf_level=CONF_LEVEL,
):
    # ---------- READ CREDENTIALS FROM ENV ----------
    DB_HOST = os.getenv("DB_HOST")
    DB_PORT = os.getenv("DB_PORT", "5432")
    DB_NAME = os.getenv("DB_NAME")
    DB_USER = os.getenv("DB_USER")
    DB_PASS = os.getenv("DB_PASSWORD")

    if not all([DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASS]):
        raise RuntimeError("Missing DB credentials. Add them in Streamlit Secrets.")

    # ---------- CONNECT ----------
    engine = create_engine(
        f"postgresql+psycopg2://{DB_USER}:{DB_PASS}@{DB_HOST}:{DB_PORT}/{DB_NAME}",
        pool_pre_ping=True,
    )

    # ---------- 1) UNIVERSE ----------
    with engine.begin() as conn:
        sec = pd.read_sql(
            text("""
                SELECT symbol, name, asset_type, currency
                FROM security
                WHERE asset_type = ANY(:asset_types)
            """),
            conn,
            params={"asset_types": list(asset_types)},
        )

    if sec.empty:
        raise RuntimeError("No securities found for chosen ASSET_TYPES.")

    symbols = sec["symbol"].unique().tolist()

    # ---------- 2) PRICES ----------
    with engine.begin() as conn:
        query = """
            SELECT symbol, date, close_price
            FROM security_prices
            WHERE symbol = ANY(:syms)
        """
        if start_date:
            query += " AND date >= :start_date"
        if end_date:
            query += " AND date <= :end_date"
        query += " ORDER BY date"

        prices_raw = pd.read_sql(
            text(query),
            conn,
            params={"syms": symbols, "start_date": start_date, "end_date": end_date},
            parse_dates=["date"],
        )

    if prices_raw.empty:
        raise RuntimeError("No price history found for selected filters.")

    prices = prices_raw.pivot(index="date", columns="symbol", values="close_price")
    prices = prices.sort_index().ffill().dropna(how="any")

    # ---------- Remove assets with insufficient history ----------
    enough = prices.count() >= min_history_days
    prices = prices.loc[:, enough]

    if prices.shape[1] == 0:
        raise RuntimeError(f"No assets with at least {min_history_days} valid days.")

    # ---------- 3) RETURNS ----------
    returns = prices.pct_change().dropna()
    mu = returns.mean().values * 252
    Sigma = np.cov(returns.values, rowvar=False) * 252
    n = len(mu)

    # ---------- 4) OPTIMIZATION ----------
    w = cp.Variable(n)
    objective = cp.Maximize(mu @ w - 5.0 * cp.quad_form(w, Sigma))
    constraints = [
    cp.sum(w) == 1,
    w >= 0]


    prob = cp.Problem(objective, constraints)
    prob.solve(solver=cp.ECOS)

    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"Optimization failed: {prob.status}")

    weights = np.maximum(w.value, 0)
    weights = weights / weights.sum()
    w_series = pd.Series(weights, index=returns.columns)

    # ---------- 5) PORTFOLIO DAILY RETURNS ----------
    port_daily = (returns * w_series).sum(axis=1)

    return port_daily, w_series

