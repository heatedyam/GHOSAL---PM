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

    # IMPORTANT NOTE: INVARIANT, NO KNOWN WAY TO GET MARKET CAPS AT VARYING POINTS IN TIME
    for symbol in symbols_list:
        ticker = yf.Ticker(symbol)
        mkt_caps[symbol] = ticker.info['marketCap']

    return tracking_dict, sector_dict, view_dict, confidences, mkt_caps



def fetch_pitched_stock_data(tracking_dict, fetch_data_dates, currency_dict):
    print(tracking_dict.keys())

    # getting market data from the first pitch date to later fit the date as a column
    # in our own dataframe
    # ==== MARKET PRICES SHOULD ONLY GO UP TO THE END OF THIS ITERATION'S DATES ====
    market_prices = yf.download('SPY', start=fetch_data_dates[0], end=fetch_data_dates[-1])['Adj Close']

    # MAKE A DATABASE OF THIS IN MYSQL ALREADY
    conversion_df = pd.read_csv('conversion_table.csv', header=0, index_col=0)


    # initialising empty master dataframe with dates from market_prices as a column
    # the column 'Date' will later be changed to an index
    merged_prices = pd.DataFrame(market_prices.index.values, columns=['Date'])

    # getting price data for all stocks in tracking_dict, tracking from pitch date.
    for date in fetch_data_dates[:-1]:

        # a 'batch' comprises stocks pitched on a specific day of pitches.

        # creating a 'batch': string of tickers for the current date
        batch = ''.join(key + ' ' for key in tracking_dict if tracking_dict[key] == date)
        print(f'Batch: {batch}')

        # fetching historical price data for the batch
        # logging.info(f'Batch: {batch}, date: {date}')
        price = yf.download(batch, start=date, end=fetch_data_dates[-1])['Adj Close']


        # APPLYING CURRENCY CONVERSION
        # headers = price.columns
        # for symbol in headers:
        #     target_currency = currency_dict[symbol]
        #     exchange_rate = conversion_df[conversion_df.index == date][target_currency].item()
        #     print(f'EXCH. RATE: {exchange_rate}')


        # yfinance returns a series if it only downloads from 1 ticker
        # so this covers that by turning the series into a dataframe
        if isinstance(price, pd.Series):
            price_df = pd.DataFrame(index=price.index)
            # price_df['Date'] = price.index
            price_df[batch.strip()] = price.values
            price = price_df


        # replaces default date index from yf with regular integer index
        price.reset_index(inplace=True, drop=False)

        # merging batch data into the master dataframe
        merged_prices = pd.merge(merged_prices, price, on=['Date'], how='left')

        # turning ['Date'] column into index
        merged_prices.set_index('Date', inplace=True)

    print('MERGED PRICES')
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
    print(f'COV MATRIX:')
    print(cov_matrix)
    print(f"mkt_weights shape: {mkt_weights.shape}")
    print(mkt_weights)
    print(f'Type of mkt_weights: {type(mkt_weights)}')

    return ((risk_aversion * cov_matrix * (mkt_weights)) + risk_free_rate)


def optimisation(merged_prices, confidences, view_dict, mkt_caps, capital, end_date):
    # OPTIMISATION
    # FEEDING WRONG LATEST_PRICES
    # KEEPS USING THE SAME PRICES FOR EVERY
    # ITERATION
    # INSTEAD, SINCE WE'RE ITERATING
    # WITH THE DATES,
    # USE THE MERGED_PRICES ROW AT THAT DATE
    # YOU'RE ITERATING IN, MULTIPLY BY WEIGHT
    # AND SUM
    symbols_list = [symbol for symbol, view in view_dict.items()]
    print(f'SYMBOLS: {symbols_list}')
    print(f'VIEW DICT: {view_dict}')
    # current US 10-year treasury yield
    risk_free_rate = 0.0414
    latest_prices = merged_prices.iloc[-1]
    returns = merged_prices.pct_change().dropna()


    mu = expected_returns.mean_historical_return(merged_prices)
    Sigma = risk_models.sample_cov(merged_prices)
    # constructing market prior
    # END SHOULD BE THE LOOP ITERATION DATE WE ARE OPTIMISING FOR
    spy_prices = yf.download('SPY', start=START_DATE, end=end_date)['Adj Close']
    delta = black_litterman.market_implied_risk_aversion(spy_prices)

    print(Sigma)
    print(mkt_caps)

    # if len(view_dict) == 1:
    #     prior = one_asset_prior(
    #         mkt_caps_dict = mkt_caps,
    #         risk_aversion=delta,
    #         cov_matrix=Sigma,
    #         risk_free_rate=risk_free_rate
    #     )
    #     print(f'Prior: {prior}')
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
    print(f'VIEW DICT: {view_dict}')
    bl = BlackLittermanModel(
        Sigma,
        tickers=symbols_list,
        absolute_views=view_dict,
        view_confidences=confidences,
        pi=prior,
        omega='idzorek'
    )
    rets = bl.bl_returns()
    ef = EfficientFrontier(rets, Sigma, weight_bounds=(lower_bound, upper_bound))
    # obj. func. --> max sharpe. other options exist too.
    ef_weights = ef.max_sharpe()

    # Discrete allocation - start with 15000 -
    # when you download Adj Close between t and t + 1, it'll get price at t
    da = DiscreteAllocation(ef_weights, latest_prices, total_portfolio_value=capital, short_ratio=None)
    print(f'EF WEIGHTS:')
    print(ef_weights)

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

        # # converting to target currency
        exchange_rate = convert_currency(
            symbol=symbol,
            date=date,
            currency_dict=currency_dict,
            # target_currency='USD'
        )
        print(f'Exchange rate: {exchange_rate}')
        position_value = price * qty
        print(f'BASE CURRENCY POSITION VALUE - {symbol}: {position_value}')
        # try:
        #     converted_position_value = position_value * exchange_rate
        #     value_dict[symbol] = converted_position_value
        #     portfolio_value += converted_position_value
        #     print(f'CONVERTED CURRENCY POSITION VALUE - {symbol}: {converted_position_value}')
        # except:
        value_dict[symbol] = position_value
        portfolio_value += position_value


    logging.info(f'Portfolio value: ${portfolio_value}')

    for symbol, value in value_dict.items():
        print(f'{symbol} value: {value}')

    return portfolio_value


def plot_data(data, subject):

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


def convert_currency(symbol, date, currency_dict, target_currency='USD'):
    print(f'date: {date}')
    date_object = datetime.strptime(date, '%Y-%m-%d').date()

    c = CurrencyRates()
    base_currency = currency_dict[symbol]
    # changed
    if base_currency != target_currency:
        # worth checking if currency is wrong way round
        exchange_rate = c.get_rate(base_currency, target_currency, date_object)
        logging.info(f'{base_currency} --> {target_currency} = {exchange_rate}')
    else:
        exchange_rate = 1

    print(f'{symbol} CURRENCY: {base_currency}, EXCH RATE: {exchange_rate}')
    return exchange_rate


def make_conversion_table(currency_dict, dates, target_currency='USD'):

    symbols = [symbol for symbol, currency in currency_dict.items()]
    unique_currencies = sorted(list(set([currency for symbol, currency in currency_dict.items()])))

    unique_pairs = {}
    for symbol, currency in currency_dict.items():
        if currency not in unique_pairs:
            unique_pairs[symbol] = currency

    master_df = pd.DataFrame(index=dates, columns=unique_currencies)
    for symbol, currency in unique_pairs.items():
        print(f'{symbol}, {currency}')
        table = []
        for date in dates:
            value = 1
            exchange_rate = convert_currency(
                symbol=symbol,
                date=date,
                currency_dict=currency_dict
            )
            table.append(exchange_rate)
        master_df[currency] = table
    print('MASTER CURRENCY DF')
    print(master_df)
    return master_df


def main():

    # INITIALISING .CSV AS DF
    # recursion requires cutting entire rows out, which would be more convenient with a pd.df
    # likely going to have to make a pd.df to
    # columns: Symbol, Pitch date, Sector, View, Confidence
    # csv_df = pd.read_csv('pitched_stock_info.csv', header=0, index_col=False)
    csv_df = pd.read_csv('mgmt_picks.csv', header=0, index_col=False)
    # need plots for:
    # 1. strategically optimised
    # 2. optimised all stocks
    # 3. equal weighted portfolio
    # 4. benchmark

    print(csv_df)
    # getting all dates
    dates = sorted(list(set(csv_df['Pitch date'].tolist())))
    logging.info(f'All dates: {dates}')

    # for currency conversion of positions
    currency_dict = {}
    for symbol in csv_df['Symbol'].tolist():
        ticker = yf.Ticker(symbol)
        currency_dict[symbol] = ticker.info['currency'].upper()

    # currency_table = make_conversion_table(
    #     currency_dict=currency_dict,
    #     dates=dates,
    #     target_currency='USD'
    # )
    print(currency_table)


    #RECURSIVE OPTIMISATION
    #TEMP VARIABLES

    # FIRST LOOP WILL START WITH INITIAL CAPITAL OF 15000
    # 15084.63
    capital_values = [15000]
    portfolio_values = []
    for index, date in enumerate(dates):
        # if index == 0:
        #     continue

        logging.info(f'Iteration for {date}')
        logging.info(f'Available capital: {capital_values[index]}')

        # filtering out symbols and dates that are not in the time period we're looking at in
        # this iteration
        # REPLACE WITH CSV_DF DROP
        loop_dates = [day for index, day in enumerate(dates) if index <= dates.index(date)]
        # list slicing; probably much faster
        # loop_dates = dates[:index+1]


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

        # +2 bc slicing does not include the element after :, and we want the element after the index, so we add 2
        fetch_data_dates = dates[:index+2]
        merged_prices = fetch_pitched_stock_data(
            tracking_dict=tracking_dict,
            fetch_data_dates=fetch_data_dates,
            currency_dict=currency_dict
        )
        print(merged_prices)


        # headers = merged_prices.columns
        # for symbol in headers:
        #     symbol_prices = merged_prices[symbol]
        #     for price in symbol_prices:


        # LOOP-BREAK CONDITION
        if 0 <= (index + 1) < len(dates):
            # CHANGED date --> dates[index + 1] as we will assume that we liquidate and repurchase 7 days from now
            # MOVING WITHIN LOOPBREAK-CONDITIONAL TO PREVENT INDEX ERROR
            allocation, leftover = optimisation(merged_prices, confidences, view_dict, mkt_caps, capital_values[index], dates[index + 1])
            # assuming that we're going to liquidate and repurchase on the same day 7 days from now

            portfolio_value = get_portfolio_value(allocation, dates[index + 1], currency_dict)

            portfolio_values.append(portfolio_value)

            new_capital_value = round((portfolio_value + leftover), 2)
            print(f'portfolio_value: {portfolio_value}')
            capital_values.append(new_capital_value)
        else:
            break

    portfolio_tracker_df = pd.DataFrame(index=dates)
    portfolio_tracker_df['Value'] = capital_values
    plot_data(data=portfolio_tracker_df, subject='Black-Litterman')

    print(capital_values)
    for index, value in enumerate(capital_values):
        logging.info(f'capital_value at {dates[index]}: {value}')


if __name__ == '__main__':
    main()
