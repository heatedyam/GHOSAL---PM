import numpy as np
import pandas as pd
import yfinance as yf
import seaborn as sns


from datetime import datetime
from dateutil.relativedelta import relativedelta
from matplotlib import pyplot as plt
from pypfopt import plotting

from pypfopt import (
    CLA,
    BlackLittermanModel,
    EfficientFrontier,
    HRPOpt,
    black_litterman,
    expected_returns,
    plotting,
    risk_models,
)

# Reading in the data; preparing expected returns and a risk model
df = pd.read_csv("tests/resources/stock_prices.csv", parse_dates=True, index_col="date")
returns = df.pct_change().dropna()
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)


# Now try with a nonconvex objective from  Kolm et al (2014)
# def deviation_risk_parity(w, cov_matrix):
#     diff = w * np.dot(cov_matrix, w) - (w * np.dot(cov_matrix, w)).reshape(-1, 1)
#     return (diff**2).sum().sum()


# ef = EfficientFrontier(mu, S)
# weights = ef.nonconvex_objective(deviation_risk_parity, ef.cov_matrix)
# ef.portfolio_performance(verbose=True)

"""
Expected annual return: 22.9%
Annual volatility: 19.2%
Sharpe Ratio: 1.09
"""

# Black-Litterman
spy_prices = pd.read_csv(
    "tests/resources/spy_prices.csv", parse_dates=True, index_col=0, squeeze=True
)
delta = black_litterman.market_implied_risk_aversion(spy_prices)

mcaps = {
    "GOOG": 927e9,
    "AAPL": 1.19e12,
    "FB": 574e9,
    "BABA": 533e9,
    "AMZN": 867e9,
    "GE": 96e9,
    "AMD": 43e9,
    "WMT": 339e9,
    "BAC": 301e9,
    "GM": 51e9,
    "T": 61e9,
    "UAA": 78e9,
    "SHLD": 0,
    "XOM": 295e9,
    "RRC": 1e9,
    "BBY": 22e9,
    "MA": 288e9,
    "PFE": 212e9,
    "JPM": 422e9,
    "SBUX": 102e9,
}
prior = black_litterman.market_implied_prior_returns(mcaps, delta, S)

# 1. SBUX will drop by 20%
# 2. GOOG outperforms FB by 10%
# 3. BAC and JPM will outperform T and GE by 15%
views = np.array([-0.20, 0.10, 0.15]).reshape(-1, 1)
picking = np.array(
    [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
        [1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, -0.5, 0, 0, 0.5, 0, -0.5, 0, 0, 0, 0, 0, 0, 0, 0.5, 0],
    ]
)
bl = BlackLittermanModel(S, Q=views, P=picking, pi=prior, tau=0.01)
rets = bl.bl_returns()
ef = EfficientFrontier(rets, S)
ef.max_sharpe()
print(ef.clean_weights())
ef.portfolio_performance(verbose=True)
