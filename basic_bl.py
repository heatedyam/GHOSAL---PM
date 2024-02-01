import logging

import csv
from datetime import datetime
from dateutil.relativedelta import relativedelta
from forex_python.converter import CurrencyRates
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd
import plotly
import seaborn as sns
import yfinance as yf
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
  
'''

# we want to get the market caps at the date each stock was pitched. postponed for now
    # for symbol in symbols:
    #     caps_df = yf.download(symbol, start=date, end=date)['marketCap']
    #     mcaps[symbol] = caps_df


# global constant.
global TODAY_DATE
TODAY_DATE = (datetime.today()).strftime('%Y-%m-%d')

# how far back we'll look for SPY prices (for df fitting purposes & mkt. prior)
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
        # confidences will be converted to numpy array
        confidence_list = []
        mkt_caps = {}
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
            mkt_caps[line[0]] = ticker.fast_info['marketCap']

        confidences = np.array(confidence_list)
        return tracking_dict, sector_dict, view_dict, confidences, mkt_caps


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
    print(tracking_dict.keys())


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
        print(f'Batch: {batch}')
        # fetching historical price data for the batch
        # logging.info(f'Batch: {batch}, date: {date}')
        price = yf.download(batch, start=date, end=TODAY_DATE)['Adj Close']

        # yfinance returns a series if it only downloads from 1 ticker
        # so this covers that by turning the series into a dataframe
        if isinstance(price, pd.Series):
            price_df = pd.DataFrame(index=price.index)
            # price_df['Date'] = price.index
            price_df[batch] = price.values
            price = price_df


        # replaces default date index from yf with regular integer index
        price.reset_index(inplace=True, drop=False)

        # merging batch data into the master dataframe
        merged_prices = pd.merge(merged_prices, price, on=['Date'], how='left')

        # turning ['Date'] column into index
        merged_prices.set_index('Date', inplace=True)

    print(merged_prices)

    # returns Pandas dataframe
    return merged_prices


# black_litterman.py seems to freak out when we only give it 1 asset.
# to isolate risk of runtime error, creating this func instead
# ERROR: results in a 2x1 matrix
def one_asset_prior(mkt_caps_dict, risk_aversion, cov_matrix, risk_free_rate):

    mkt_caps = pd.Series(mkt_caps_dict)
    mkt_weights = mkt_caps / mkt_caps.sum()

    # ADDED BY ELMO - due to dot product issues with mkt_weights
    # Pi is excess returns so must add risk_free_rate to get return.
    print(f'COV MATRIX: {cov_matrix}')
    print(f"mkt_weights shape: {mkt_weights.shape}")
    print(mkt_weights)
    print(f'Type of mkt_weights: {type(mkt_weights)}')

    return ((risk_aversion * cov_matrix * (mkt_weights)) + risk_free_rate)


def optimisation(merged_prices, confidences, view_dict, mkt_caps, capital):
    # OPTIMISATION
    print(f'VIEW DICT: {view_dict}')
    # current US 10-year treasury yield
    risk_free_rate = 0.0414
    latest_prices = merged_prices.iloc[-1]
    returns = merged_prices.pct_change().dropna()

    mu = expected_returns.mean_historical_return(merged_prices)
    Sigma = risk_models.sample_cov(merged_prices)
    # constructing market prior
    spy_prices = yf.download('SPY', start=START_DATE, end=TODAY_DATE)['Adj Close']
    delta = black_litterman.market_implied_risk_aversion(spy_prices)

    print(Sigma)
    print(mkt_caps)

    # if len(view_dict) == 1:
    #     # prior = one_asset_prior(
    #     #     mkt_caps_dict = mkt_caps,
    #     #     risk_aversion=delta,
    #     #     cov_matrix=Sigma,
    #     #     risk_free_rate=risk_free_rate
    #     # )
    #     # print(f'Prior: {prior}')
    #     pass
    # else:

    # ----> VALUEERROR: matrices are not aligned
    #   cov_matrix.dot(mkt_weights) + risk_free_rate
    prior = black_litterman.market_implied_prior_returns(
        market_caps=mkt_caps,
        risk_aversion=delta,
        cov_matrix=Sigma,
        risk_free_rate=risk_free_rate
    )
    print(f'Prior: {prior}')
    # logging.info(f'Prior: {prior}')

    # setting weight bounds --> to deviate 5% from a 1/n portfolio, given that 1/n portfolios have strong returns
    deviation = 0.05
    #
    bounds = float(1/len(confidences))
    lower_bound = max(0, (bounds - deviation))
    upper_bound = bounds + deviation

    # Black-Litterman optimisation
    bl = BlackLittermanModel(Sigma, absolute_views=view_dict, view_confidences=confidences, pi=prior, omega='idzorek')
    rets = bl.bl_returns()
    ef = EfficientFrontier(rets, Sigma, weight_bounds=(lower_bound, upper_bound))
    # obj. func. --> max sharpe. other options exist too.
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

    # 23rd nov: thanksgiving, so mkts were closed. crude fix, will probably make a list and add to it as we go along.
    if date == '2023-11-23':
        date = '2023-11-24'

    portfolio_value = 0
    gbp_portfolio_value = 0
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

        price = round(float(df.values), 2)
        logging.info(f'{symbol} price: {price} {currency_dict[symbol]}')

        # converting to GBP
        c = CurrencyRates()
        base_currency = currency_dict[symbol]
        target_currency = 'GBP'
        if base_currency != target_currency:
            exchange_rate = c.get_rate(base_currency, target_currency, start_date)
            logging.info(f'{base_currency} --> {target_currency} = {exchange_rate}')
        else:
            exchange_rate = 1

        gbp_price = price * exchange_rate
        position_value = price * qty
        gbp_position_value = position_value * exchange_rate

        value_dict[symbol] = position_value
        # logging.info(f'{symbol} position value: {position_value} {currency_dict[symbol]}')
        logging.info(f'{symbol} position GBP value: £{gbp_position_value}')

        portfolio_value += position_value
        gbp_position_value += gbp_position_value

    logging.info(f'Portfolio value: ${portfolio_value}')
    logging.info(f'GBP portfolio value: ${gbp_portfolio_value}')

    for symbol, value in value_dict.items():
        print(f'{symbol} value: {value} {currency_dict[symbol]}')

    return portfolio_value


def write_plot(data, subject):

    file_name = f'{TODAY_DATE} - MGMT-picked Discretionary Portfolio - Value'

    # working directory is same as the script's
    path = f'Graphs/{TODAY_DATE}/{subject}'

    # LEGACY - SEABORN PLOTTING
    plt.figure(figsize=(14, 8))
    sns.set(style="darkgrid")

    for column in data.columns:
        sns.lineplot(x=data.index, y=data[column], label=column)

    plt.title(f'Discretionary portfolio - value over time')
    plt.xlabel('Date')
    plt.ylabel('Value (£)')
    plt.legend(loc='upper right', bbox_to_anchor=(1.12, 1.00))


    # if defined path doesn't exist, it'll make it for you.
    if not os.path.exists(path):
        # use makedirs for nested directories
        os.makedirs(path)
        # saves all sector plots with unique names
        plt.savefig(f'{path}/{file_name}.png')

    else:
        plt.savefig(f'{path}/{file_name}.png')


def main():

    # INITIALISING .CSV AS DF
    # recursion requires cutting entire rows out, which would be more convenient with a pd.df
    # likely going to have to make a pd.df to
    # columns: Symbol, Pitch date, Sector, View, Confidence
    csv_df = pd.read_csv('mgmt_picks.csv', header=0, index_col=False)

    print(csv_df)

    # for currency conversion of positions
    currency_dict = {}
    for symbol in csv_df['Symbol'].tolist():
        ticker = yf.Ticker(symbol)
        currency_dict[symbol] = ticker.info['currency'].upper()


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
        print(merged_prices)

        allocation, leftover = optimisation(merged_prices, confidences, view_dict, mkt_caps, capital_values[index])
        # assuming that we're going to liquidate and repurchase on the same day 7 days from now
        if 0 <= (index + 1) < len(dates):
            portfolio_value = get_portfolio_value(allocation, dates[index + 1], currency_dict)

            portfolio_values.append(portfolio_value)

            new_capital_value = round((portfolio_value + leftover), 2)
            print(f'portfolio_value: {portfolio_value}')
            capital_values.append(new_capital_value)
        else:
            break

    portfolio_tracker_df = pd.DataFrame(index=dates)
    portfolio_tracker_df['Value'] = capital_values
    write_plot(data=portfolio_tracker_df, subject='Black-Litterman')

    print(capital_values)
    for index, value in enumerate(capital_values):
        logging.info(f'capital_value at {dates[index]}: £{value}')


if __name__ == '__main__':
    main()
