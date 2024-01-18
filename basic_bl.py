import logging

from forex_python.converter import CurrencyRates
import numpy as np
import pandas as pd
import yfinance as yf
import seaborn as sns
import csv
from datetime import datetime
from dateutil.relativedelta import relativedelta
from matplotlib import pyplot as plt
from pypfopt import (
    CLA,
    BlackLittermanModel,
    EfficientFrontier,
    HRPOpt,
    black_litterman,
    expected_returns,
    plotting,
    risk_models,
    DiscreteAllocation,
    plotting
)

# setting up logging module for debugging purposes
logging.basicConfig(
    # filename='get data - yfinance.log',
    # levels in ascending order : DEBUG, INFO, WARNING, ERROR, CRITICAL
    # everything below a level will not be printed in terminal at runtime
    level=logging.INFO,
    format='[%(levelname)s] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

'''
To resolve:
-parsing list of symbols from tracking_dict
-resolving staggered stock price data into black-litterman model views
-create function that only includes stocks in tracking_dict before today 

-start with 15K at start
-at the pitch dates, multiply discrete allocation by closing price of that day
-and run optimisation again using the old stocks as well as the new stocks pitched
for the new dates
  
'''

# we want to get the market caps at the date each stock was pitched. postponed for now
    # for symbol in symbols:
    #     caps_df = yf.download(symbol, start=date, end=date)['marketCap']
    #     mcaps[symbol] = caps_df


# global constant.
global TODAY_DATE
TODAY_DATE = (datetime.today()).strftime('%Y-%m-%d')

# how far back we'll look for SPY prices (for df fitting purposes)
years_back = 3
today_date_raw = datetime.today()
start_date_raw = today_date_raw - relativedelta(years=years_back)

global START_DATE
START_DATE = start_date_raw.strftime('%Y-%m-%d')


def csv_to_dicts():
    # reading csv file and parsing contents into dictionaries
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
            # views
            view_dict[line[0]] = float(line[3])
            # confidences --> to numpy array
            confidence_list.append(float(line[4]))

            # market caps
            ticker = yf.Ticker(line[0])
            mcaps[line[0]] = ticker.fast_info['marketCap']

        confidences = np.array(confidence_list)
        return tracking_dict, sector_dict, view_dict, confidences, mcaps


def df_to_dicts(csv_df):
    tracking_dict = dict(zip(csv_df['Symbol'], csv_df['Pitch date']))
    sector_dict = dict(zip(csv_df['Symbol'], csv_df['Sector']))
    view_dict = dict(zip(csv_df['Symbol'], csv_df['View']))

    confidence_list = csv_df['Confidence'].tolist()
    confidences = np.array(confidence_list)

    mkt_caps = {}
    symbols_list = csv_df['Symbol'].tolist()
    # symbols_str = ''.join(symbol + ' ' for symbol in df['Symbol'].values)

    for symbol in symbols_list:
        ticker = yf.Ticker(symbol)
        mkt_caps[symbol] = ticker.info['marketCap']

    return tracking_dict, sector_dict, view_dict, confidences, mkt_caps


def fetch_pitched_stock_data(tracking_dict, dates):
    # getting market data from the first pitch date to later fit the date as a column
    # in our own dataframe
    market_prices = yf.download("SPY", start=dates[0], end=TODAY_DATE)["Adj Close"]

    # initialising empty master dataframe with dates from market_prices as a column
    # the column 'Date' will later be changed to an index
    merged_prices = pd.DataFrame(market_prices.index.values, columns=['Date'])

    # getting price data for all stocks in tracking_dict, tracking from pitch date.
    for date in dates:
        # a 'batch' comprises stocks pitched on a specific day of pitches.

        # creating a 'batch': string of tickers for the current date
        batch = ''.join(key + ' ' for key in tracking_dict if tracking_dict[key] == date)

        # fetching historical price data for the batch
        # logging.info(f'Batch: {batch}, date: {date}')
        price = yf.download(batch, start=date, end=TODAY_DATE)['Adj Close']
        # replaces default date index from yf with regular integer index
        price.reset_index(inplace=True, drop=False)

        # merging batch data into the master dataframe
        merged_prices = pd.merge(merged_prices, price, on=['Date'], how='left')

        # turning ['Date'] column into index
        merged_prices.set_index('Date', inplace=True)

    # returns Pandas dataframe
    return merged_prices


def optimisation(merged_prices, confidences, view_dict, mkt_caps, capital):
    # OPTIMISATION
    latest_prices = merged_prices.iloc[-1]
    returns = merged_prices.pct_change().dropna()

    mu = expected_returns.mean_historical_return(merged_prices)
    Sigma = risk_models.sample_cov(merged_prices)

    # constructing market prior
    spy_prices = yf.download('SPY', start=START_DATE, end=TODAY_DATE)['Adj Close']
    delta = black_litterman.market_implied_risk_aversion(spy_prices)
    prior = black_litterman.market_implied_prior_returns(mkt_caps, delta, Sigma)
    # logging.info(f'Prior: {prior}')

    # setting weight bounds --> to deviate 5% from a 1/n portfolio, given that a 1/n portfolio
    # has strong returns
    deviation = 0.05
    bounds = (1/len(confidences))
    lower_bound = max(0, (bounds - deviation))
    upper_bound = bounds + deviation

    # Black Litterman optimisation
    bl = BlackLittermanModel(Sigma, absolute_views=view_dict, view_confidences=confidences, pi=prior, omega='idzorek')
    rets = bl.bl_returns()
    ef = EfficientFrontier(rets, Sigma, weight_bounds=(lower_bound, upper_bound))
    ef_weights = ef.max_sharpe()

    # Discrete allocation - start with 15000 -
    # when you download Adj Close between t and t + 1, it'll get price at t
    da = DiscreteAllocation(ef_weights, latest_prices, total_portfolio_value=capital, short_ratio=None)

    # OrderedDict
    # for DiscreteAllocation, at the end of every period, calc. current value of positions
    # and enter that into DiscreteAllocation constraint
    ef.portfolio_performance(verbose=True)

    allocation, leftover = da.greedy_portfolio()
    print(f"Discrete allocation (no. shares): {allocation}")
    print(f"Funds remaining: ${leftover:.2f}")

    return allocation, leftover


def get_portfolio_value(allocation, date, currency_dict):

    # tried to accommodate for closed trading day - only fix that seems to work
    if date == '2023-11-23':
        date = '2023-11-24'

    portfolio_value = 0
    value_dict = {}
    # gets adjacent day 1 day ahead
    # .strptime converts a date string into a date object given a specific format
    start_date = (pd.to_datetime(date))
    adj_date_raw = datetime.strptime(date,'%Y-%m-%d') + relativedelta(days=1)
    adj_date = adj_date_raw.strftime('%Y-%m-%d')

    for symbol, qty in allocation.items():
        logging.info(f'{symbol} qty: {qty}')

        # exception handling in the case that we try to get price data
        # on a day where markets are closed
        try:
            logging.info(f'Start date: {date}')
            logging.info(f'End date: {adj_date}')
            df = yf.download(symbol, start=date, end=adj_date)['Adj Close']
            print(df)

        except Exception:
            print('EXCEPTION RAISED')
            logging.info(f'TRYING AGAIN FOR {symbol}')

            new_start_date = (pd.to_datetime(date) - pd.DateOffset(days=2)).strftime('%Y-%m-%d')
            new_adj_date = (pd.to_datetime(new_start_date) + pd.DateOffset(days=1)).strftime('%Y-%m-%d')

            logging.info(f'Exception Start date: {new_start_date}')
            logging.info(f'Exception End date: {new_adj_date}')

            df = yf.download(symbol, start=new_start_date, end=new_adj_date)['Adj Close']
            print(df)


        logging.info(f'Getting position value for: {symbol}')
        # NO COLUMN NAME - DF[SYMBOL] DOESN'T WORK


        price = float(df.values)
        logging.info(f'{symbol} price: {price} {currency_dict[symbol]}')
        # converting to GBP
        # c = CurrencyRates()
        # base_currency = currency_dict[symbol]
        # target_currency = 'GBP'
        # if base_currency != target_currency:
        #     exchange_rate = c.get_rate(base_currency, target_currency, start_date)
        #     logging.info(f'{base_currency} --> {target_currency} = {exchange_rate}')
        # else:
        #     exchange_rate = 1

        # gbp_price = price * exchange_rate
        position_value = price * qty

        value_dict[symbol] = position_value
        logging.info(f'{symbol} position value: {position_value} {currency_dict[symbol]}')

        portfolio_value += position_value

    logging.info(f'Portfolio value: ${portfolio_value}')
    for symbol, value in value_dict.items():
        print(f'{symbol} value: ${value}')

    return portfolio_value


def main():

    # INITIALISING .CSV AS DF
    # recursion requires cutting entire rows out, which would be more convenient with a pd.df
    # likely going to have to make a pd.df to
    csv_df = pd.read_csv('pitched_stock_info.csv', header=None, index_col=False)
    logging.info('CSV_DF - PRE-COLUMNS')
    print(csv_df)
    columns = ['Symbol', 'Pitch date', 'Sector', 'View', 'Confidence']
    csv_df.columns = columns
    logging.info('CSV_DF - POST-COLUMNS')
    print(csv_df)

    currency_dict = {}
    for symbol in csv_df['Symbol'].tolist():
        ticker = yf.Ticker(symbol)
        currency_dict[symbol] = ticker.info['currency'].upper()




    # INITIALISING DICTS/LISTS

    # getting all dates
    dates = sorted(list(set(csv_df['Pitch date'].tolist())))
    logging.info(f'All dates: {dates}')

    #RECURSIVE OPTIMISATION
    #TEMP VARIABLES

    # FIRST LOOP WILL START WITH INITIAL CAPITAL OF 15000
    capital_values = [15000]
    portfolio_values = []
    for index, date in enumerate(dates):

        logging.info(f'Iteration for {date}')
        logging.info(f'Available capital: ${capital_values[index]}')

        # filtering out symbols and dates that are not in the time period we're looking at in
        # this iteration
        # REPLACE WITH CSV_DF DROP
        loop_dates = [day for index, day in enumerate(dates) if index <= dates.index(date)]
        logging.info(f'loop_dates: {loop_dates}')

        # CSV_DF FILTERING - ONLY STOCKS PITCHED WITHIN THIS ITERATION'S DATES
        filtered_csv_df = csv_df[csv_df['Pitch date'].isin(loop_dates)]
        logging.info('FILTERED .CSV DF')
        print(filtered_csv_df)

        tracking_dict, sector_dict, view_dict, confidences, mkt_caps = df_to_dicts(filtered_csv_df)
        logging.info(f'tracking_dict: {tracking_dict}')
        # logging.info(f'sector_dict: {sector_dict}')
        # logging.info(f'view_dict: {view_dict}')
        # logging.info(confidences)
        # logging.info(f'mkt_caps: {mkt_caps}')

        merged_prices = fetch_pitched_stock_data(tracking_dict, loop_dates)
        print(merged_prices.head())

        # CONVERT THE .CSV INTO A DF AND DELETE ROWS AS NEEDED
        allocation, leftover = optimisation(merged_prices, confidences, view_dict, mkt_caps, capital_values[index])
        # assuming that we're going to liquidate and repurchase on the same day 7 days from now
        if 0 <= (index + 1) < len(dates):
            portfolio_value = get_portfolio_value(allocation, dates[index + 1], currency_dict)

            portfolio_values.append(portfolio_value)

            new_capital_value = portfolio_value + leftover
            print(f'portfolio_value: {portfolio_value}')
            capital_values.append(new_capital_value)
        else:
            break

    for index, value in enumerate(capital_values):
        print(f'capital_value at {dates[index]}: ${value}')


if __name__ == '__main__':
    main()


#

