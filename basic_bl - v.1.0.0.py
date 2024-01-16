import numpy as np
import pandas as pd
import yfinance as yf
import seaborn as sns
import csv

from datetime import datetime
from dateutil.relativedelta import relativedelta
from matplotlib import pyplot as plt
from pypfopt import plotting

'''
To resolve:
-parsing list of symbols from tracking_dict
-resolving staggered stock price data into black-litterman model views
-create function that only includes stocks in tracking_dict before today 

'''


from pypfopt import (
    CLA,
    BlackLittermanModel,
    EfficientFrontier,
    HRPOpt,
    black_litterman,
    expected_returns,
    plotting,
    risk_models,
    DiscreteAllocation
)

years_back = 3

today_date_raw = datetime.today()
start_date_raw = today_date_raw - relativedelta(years=years_back)
today_date = today_date_raw.strftime('%Y-%m-%d')
start_date = start_date_raw.strftime('%Y-%m-%d')




with open('pitched_stock_info.csv', 'r') as f:
    tracking_dict = {}
    sector_dict = {}
    view_dict = {}
    #SHOULD BE NUMPY ARRAY, NOT LIST
    confidence_list = []
    mcaps = {}
    reader = csv.reader(f)
    for line in reader:

        # key: symbol --> value: date pitched
        tracking_dict[line[0]] = line[1]
        # key: symbol --> value: stock sector
        sector_dict[line[0]] = line[2]
        view_dict[line[0]] = float(line[3])
        confidence_list.append(float(line[4]))

        ticker = yf.Ticker(line[0])
        mcaps[line[0]] = ticker.fast_info['marketCap']
    confidences = np.array(confidence_list)

dates = sorted(list(set(tracking_dict.values())))

symbols_str = ''.join(key + ' ' for key in tracking_dict)
# merged_prices = yf.download(symbols_str, start=start_date, end=today_date)['Adj Close']


# getting market data from the first pitch date to later fit the date as a column
# in our own dataframe
market_prices = yf.download("SPY", start=dates[0], end=today_date)["Adj Close"]

# initialising empty master dataframe with dates from market_prices as a column
# the column 'Date' will later be changed to an index
merged_prices = pd.DataFrame(market_prices.index.values, columns=['Date'])

# getting price data for all stocks in tracking_dict, tracking from pitch date.
symbols = symbols_str.split(' ')

# tracking portfolio value over iterations
# method
'''
start with 15K at start
at the pitch dates, multiply discrete allocation by closing price of that day
and run optimisation again using the old stocks as well as the new stocks pitched
for the new dates
  

'''


for date in dates:
    # a 'batch' comprises stocks pitched on a specific day of pitches.

    # creating a 'batch': string of tickers for the current date
    batch = ''.join(key + ' ' for key in tracking_dict if tracking_dict[key] == date)

    # fetching historical price data for the batch
    price = yf.download(batch, start=date, end=today_date)['Adj Close']
    # replaces default date index from yf with regular integer index
    price.reset_index(inplace=True, drop=False)

    # merging batch data into the master dataframe
    merged_prices = pd.merge(merged_prices, price, on=['Date'], how='left')

    # turning ['Date'] column into index
    merged_prices.set_index('Date', inplace=True)

#
    # for symbol in symbols:
    #     caps_df = yf.download(symbol, start=date, end=date)['marketCap']
    #     mcaps[symbol] = caps_df


    #MOVING EVERYTHING INTO THE FOR LOOP

    latest_prices = merged_prices.iloc[-1]
    returns = merged_prices.pct_change().dropna()

    # Reading in the data; preparing expected returns and a risk model
    # prices = pd.read_csv("tests/resources/stock_prices.csv", parse_dates=True, index_col="date")


    mu = expected_returns.mean_historical_return(merged_prices)
    Sigma = risk_models.sample_cov(merged_prices)


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
    spy_prices = yf.download('SPY', start=start_date, end=today_date)['Adj Close']
    delta = black_litterman.market_implied_risk_aversion(spy_prices)



    prior = black_litterman.market_implied_prior_returns(mcaps, delta, Sigma)
    print(f'Prior: {prior}')

    # 1. SBUX will drop by 20%
    # 2. GOOG outperforms FB by 10%
    # 3. BAC and JPM will outperform T and GE by 15%

    # for stocks we pick
    #randomly generated views
    views = np.array([0.51, 0.61, 0.29, 0.52, 0.36, 0.26, 0.79, 0.21, 0.1, 0.38, 0.57, 0.6, 0.26, 0.14, 0.25, 0.84]).reshape(-1, 1)
    # picking = np.array(
    #     [
    #         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    #         [1, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    #         [0, 0, 0, 0, 0, -0.5, 0, 0, 0.5, 0, -0.5, 0, 0, 0, 0, 0, 0, 0, 0.5, 0],
    #     ]
    # )

    # weight bounds
    bounds = (1/len(confidences))
    lower_bound = max(0, (bounds - 0.05))
    upper_bound = bounds + 0.05

    bl = BlackLittermanModel(Sigma, absolute_views=view_dict, view_confidences=confidences, pi=prior, omega='idzorek')
    rets = bl.bl_returns()
    ef = EfficientFrontier(rets, Sigma, weight_bounds=(lower_bound, upper_bound))
    ef_weights = ef.max_sharpe()

    da = DiscreteAllocation(ef_weights, latest_prices, total_portfolio_value=15000, short_ratio=None)

    # OrderedDict
    # for DiscreteAllocation, at the end of every period, calc. current value of positions
    # and enter that into DiscreteAllocation constraint
    print(f'Cleaned weights: {ef.clean_weights()}')
    ef.portfolio_performance(verbose=True)

    allocation, leftover = da.greedy_portfolio()
    print(f"Discrete allocation (no. shares): {allocation}")
    print(f"Funds remaining: ${leftover:.2f}")


