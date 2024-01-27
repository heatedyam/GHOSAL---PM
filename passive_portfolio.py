from datetime import datetime
import csv

import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import yfinance as yf
import os
from varname import nameof



# setting up logging module for debugging purposes
logging.basicConfig(
    # filename='get data - yfinance.log',
    # levels in ascending order : DEBUG, INFO, WARNING, ERROR, CRITICAL
    # everything below a level will not be printed in terminal at runtime
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# global constant.
global TODAY_DATE
TODAY_DATE = (datetime.today()).strftime('%Y-%m-%d')

'''
Modular script to track performance of all stocks pitched.
1/n portfolio with returns tracking from date of pitch
for each stock.

Returns can be grouped by sector.

TO-DO:

-turn graphing into function w/ plotly



'''


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
        price = yf.download(batch, start=date, end=TODAY_DATE)['Adj Close']
        # replaces default date index from yf with regular integer index
        price.reset_index(inplace=True, drop=False)

        # merging batch data into the master dataframe
        merged_prices = pd.merge(merged_prices, price, on=['Date'], how='left')

        # turning ['Date'] column into index
        merged_prices.set_index('Date', inplace=True)

    # returns Pandas dataframe
    return merged_prices


# daily, avg_daily, cumul returns respectively
def get_returns(portfolio_history):
    # 1 for 1-day lookback, 21 for 1 month, 252 for 1 year
    daily_returns = portfolio_history.pct_change(1)

    # dividing daily returns by no. stocks
    avg_daily_returns = daily_returns/len(portfolio_history.columns)

    # cumulative return for all stocks. graph.
    cumul_returns = (avg_daily_returns + 1).cumprod() - 1

    return [daily_returns, avg_daily_returns, cumul_returns]


def make_dicts():
    # reading csv file and parsing contents into dictionaries
    with open('pitched_stock_info.csv', 'r') as f:
        tracking_dict = {}
        sector_dict = {}
        reader = csv.reader(f)
        for line in reader:
            # key: symbol --> value: date pitched
            tracking_dict[line[0]] = line[1]
            # key: symbol --> value: stock sector
            sector_dict[line[0]] = line[2]

    return tracking_dict, sector_dict


# groups any dataframe of returns
def group_returns_by_sector(returns, sector_dict):

    # list of sectors
    sectors = sorted(list(set(sector_dict.values())))

    # all sector returns will be appended here
    sector_return_list = []

    for sector_name in sectors:
        # list comprehension selecting stocks that are in the sector
        selected_columns = [col for col in returns.columns if sector_dict[col] == sector_name]
        logging.debug(f'{sector_name}: {selected_columns}')

        # date indexing is preserved
        sector_return = returns[selected_columns]
        # sector returns are appended to the list as identifier with accompanying dataframe
        sector_return_list.append([sector_return, sector_name])

    # returns list of df/name pairs respectively
    return sector_return_list


# should export and plot all returns
# plot daily, avg_daily, cumul returns, each for all individual stocks, and each for each sector
# parameters: single return with string containing type of data
# considering **kwargs and 'if kwargs:' conditional to handle additional parameter
# of sector dict in the case of plotting sector performance
# ** allows flexibility in filenaming
def write_plot(data, return_type, subject):

    file_name = f'{TODAY_DATE} - Passive Portfolio - {subject} - {return_type}'

    # working directory is same as the script's
    path = f'Graphs/{TODAY_DATE}/{subject}'

    # LEGACY - SEABORN PLOTTING
    plt.figure(figsize=(14, 8))
    sns.set(style="darkgrid")

    for column in data.columns:
        sns.lineplot(x=data.index, y=data[column], label=column)

    plt.title(f'Passive portfolio - {return_type} - {subject} over time')
    plt.xlabel('Date')
    plt.ylabel('Returns')
    plt.legend(loc='upper right', bbox_to_anchor=(1.12, 1.00))


    # if defined path doesn't exist, it'll make it for you.
    if not os.path.exists(path):
        # use makedirs for nested directories
        os.makedirs(path)
        # saves all sector plots with unique names
        plt.savefig(f'{path}/{file_name}.png')

    else:
        plt.savefig(f'{path}/{file_name}.png')


def aggregate_df(data):
    pass


def main():

    # uses first row of .csv as column names
    stock_info_df = pd.read_csv("pitched_stock_info.csv", header=0)

    # date format as 'YYYY-MM-DD' for yfinance compatibility
    tracking_dict, sector_dict = make_dicts()
    logging.info(f'tracking_dict: {tracking_dict}')
    logging.info(f'sector_dict: {sector_dict}')
    logging.info(print(stock_info_df.tail()))

    # so we can iterate from oldest to most recent pitches
    dates = sorted(list(set(tracking_dict.values())))
    logging.info(f'Dates: {dates}')

    # returns dataframe of historical prices for pitched stocks
    portfolio_history = fetch_pitched_stock_data(tracking_dict, dates)
    print(portfolio_history.tail())

    # INDIVIDUAL EQUITY CUMULATIVE RETURNS
    all_returns = get_returns(portfolio_history)
    # all_returns is a list that adheres to this order
    return_types = ['Daily Returns', 'Avg. Daily Returns', 'Cumul. Returns']
    for i, df in enumerate(all_returns):
        # alternate method of specifying parameters; insensitive to ordering so long as
        # there are no * or ** operators in the function's definition
        write_plot(data=df, return_type=return_types[i], subject='Individual Equity Returns')

    # 0: Daily returns, 1: avg. daily returns, 2: cumul. returns
    # returns list of df/sector name pairs respectively
    grouped_returns = group_returns_by_sector(all_returns[1], sector_dict)

    # SECTOR CUMULATIVE RETURNS
    # MAKE FUNCTION TO CREATE SINGLE AGGREGATE PLOT OF CUMUL. RETURNS?
    #   --> would be applicable to each sector, the whole portfolio, and the disc. portfolio
    avg_daily_sector_returns = pd.DataFrame(index=portfolio_history.index)
    for sector in grouped_returns:
        sector_return = sector[0]
        sector_name = sector[1]
        logging.info(f'SECTOR: {sector_name}')

        # sums all columns over the index (time)
        avg_daily_sector_return = sector_return.sum(axis=1)
        avg_daily_sector_returns[sector_name] = avg_daily_sector_return.values

    cumul_return_by_sector = (avg_daily_sector_returns + 1).cumprod() - 1
    for column in cumul_return_by_sector.columns:
        sns.lineplot(x=cumul_return_by_sector.index, y=cumul_return_by_sector[column].values, label=column)
    write_plot(data=cumul_return_by_sector, return_type=return_types[2], subject='Sector Performance')

    # PLOTTING PASSIVE PORTFOLIO VALUE OVER TIMEW
    portfolio_return_values = all_returns[1].sum(axis=1)
    portfolio_return_values = (portfolio_return_values + 1).cumprod() - 1
    # write_plot() doesn't like it when you pass in a series instead of a dataframe
    portfolio_return = pd.DataFrame(index=portfolio_history.index)
    portfolio_return['Return'] = portfolio_return_values

    write_plot(data=portfolio_return, return_type=return_types[2], subject='Portfolio Performance')


if __name__ == '__main__':
    main()
