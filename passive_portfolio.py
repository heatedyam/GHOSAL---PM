from datetime import datetime
import csv
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly
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
def get_returns(portfolio_history, tracking_dict):
    # 1 for 1-day lookback, 21 for 1 month, 252 for 1 year
    daily_returns = portfolio_history.pct_change(1)

    # dividing daily returns by no. stocks
    avg_daily_returns = daily_returns/len(tracking_dict)

    # cumulative return for all stocks. graph.
    cumul_returns = (avg_daily_returns + 1).cumprod() - 1

    return daily_returns, avg_daily_returns, cumul_returns


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

    for sector in sectors:
        # list comprehension selecting stocks that are in the sector
        selected_columns = [col for col in returns.columns if sector_dict[col] == sector]
        logging.debug(f'{sector}: {selected_columns}')

        # date indexing is preserved
        sector_return = returns[selected_columns]
        # sector returns are appended to the list as identifier with accompanying dataframe
        sector_return_list.append([sector, sector_return])

    # returns list of df/name pairs
    return sector_return_list

# export and plot all returns
# plot daily, avg_daily, cumul returns, each for all individual stocks, and each for each sector
# parameters: multiple returns
def plot_returns(returns):
    pass


def main():
    # FAULTY TICKERS
    #     HEXA-B,2023-11-30,Industrials

    stock_info_df = pd.read_csv("pitched_stock_info.csv")
    stock_info_df.columns = ['Stock symbol', 'Pitch date', 'Sector', 'View', 'Confidence']

    # date format as 'YYYY-MM-DD' for yfinance compatibility
    tracking_dict, sector_dict = make_dicts()
    logging.info(f'tracking_dict: {tracking_dict}')
    logging.info(f'sector_dict: {sector_dict}')
    logging.info(print(stock_info_df.tail()))

    # iterating from oldest to most recent pitch
    dates = sorted(list(set(tracking_dict.values())))
    logging.info(f'Dates: {dates}')

    # returns dataframe of historical prices for pitched stocks
    portfolio_history = fetch_pitched_stock_data(tracking_dict, dates)
    print(portfolio_history.tail())

    daily_returns, avg_daily_returns, cumul_returns = get_returns(portfolio_history, tracking_dict)
    return_types = ['Daily Returns', 'Avg. Daily Returns', 'Cumul. Returns']
    return_type = return_types[1]

    # stratifies returns by sector. you can select any of the 3 dataframes
    # return type will be used as part of exported filenames
    # SAVES A SET OF GROUPED RETURNS TO INDIVIDUAL SECTOR .PNG
    grouped_returns = group_returns_by_sector(avg_daily_returns, sector_dict)


    # PLOTTING ALL STOCKS ON ONE GRAPH
    plt.figure(figsize=(14, 8))
    sns.set(style="darkgrid")

    for column in cumul_returns.columns:
        sns.lineplot(x=cumul_returns.index, y=cumul_returns[column], label=column)

    plt.title(f'All stocks - {return_type} over time')
    plt.xlabel('Date')
    plt.ylabel('Returns')
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.00))


    # if folder doesn't exist, make new folder titled today's date
    if not os.path.exists(f'{TODAY_DATE} - all stocks'):
        os.mkdir(f'{TODAY_DATE} - all stocks')
        # saves all sector plots with unique names
        plt.savefig(f'{TODAY_DATE} - all stocks/passive portfolio - all stocks - {return_type} - {TODAY_DATE}.png')
    else:
        plt.savefig(f'{TODAY_DATE} - all stocks/passive portfolio - all stocks - {return_type} - {TODAY_DATE}.png')


    # PLOTTING PASSIVE PORTFOLIO VALUE OVER TIME
    portfolio_return = avg_daily_returns.sum(axis=1)
    portfolio_return = (portfolio_return + 1).cumprod() - 1

    plt.figure(figsize=(14, 8))
    sns.set(style="darkgrid")

    sns.lineplot(x=portfolio_return.index, y=portfolio_return.values, label='Passive portfolio')

    plt.title(f'Passive portfolio value over time')
    plt.xlabel('Date')
    plt.ylabel('Returns')
    plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.00))

    if not os.path.exists(f'{TODAY_DATE} - portfolio value'):
        os.mkdir(f'{TODAY_DATE} - portfolio value')
        # saves all sector plots with unique names
        plt.savefig(f'{TODAY_DATE} - portfolio value/passive portfolio - all stocks - {return_type} - {TODAY_DATE}.png')
    else:
        plt.savefig(f'{TODAY_DATE} - portfolio value/passive portfolio - all stocks - {return_type} - {TODAY_DATE}.png')





    # SECTOR RETURNS
    avg_daily_sector_returns = pd.DataFrame(index=avg_daily_returns.index)
    for returns in grouped_returns:
        sector_name = returns[0]
        df = returns[1]
        print(f'SECTOR: {sector_name}')
        # print(df.tail())

        avg_daily_sector_return = df.sum(axis=1)
        avg_daily_sector_returns[sector_name] = avg_daily_sector_return.values

    sector_cumul_return = (avg_daily_sector_returns + 1).cumprod() - 1


    # Plotting using seaborn - turn into function
    plt.figure(figsize=(10, 6))
    sns.set(style="darkgrid")

    for column in sector_cumul_return.columns:
        sns.lineplot(x=sector_cumul_return.index, y=sector_cumul_return[column].values, label=column)

    plt.title(f'Sector performance over time')
    plt.xlabel('Date')
    plt.ylabel('Returns')
    plt.legend(loc='upper left')

    # if folder doesn't exist, make new folder titled today's date
    if not os.path.exists(f'{TODAY_DATE} - sector performance'):
        os.mkdir(f'{TODAY_DATE} - sector performance')
        # saves all sector plots with unique names
        plt.savefig(f'{TODAY_DATE} - sector performance/passive portfolio - Cumulative sector returns - {TODAY_DATE}.png')
    else:
        plt.savefig(f'{TODAY_DATE} - sector performance/passive portfolio - Cumulative sector returns - {return_type} - {TODAY_DATE}.png')


if __name__ == '__main__':
    main()
