from datetime import datetime
from dateutil.relativedelta import relativedelta
import csv
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf

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
TODAY_DATE = (datetime.today()).strftime('%Y-%m-%d')

'''
Purpose: to track performance of all stocks pitched.
1/n portfolio with returns tracking from date of pitch
for each stock.

Modular script.

YET-TO-DO:

-group returns by sector --> code from brinson_main.py
-graph sector and individual returns in sns with legends
-read and initialise dicts from pitched_stock_info.csv
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


def get_returns(portfolio_history, tracking_dict):
    # 1 for 1-day lookback, 21 for 1 month, 252 for 1 year
    daily_returns = portfolio_history.pct_change(1)

    # dividing daily returns by no. stocks
    avg_daily_returns = daily_returns/len(tracking_dict)

    # cumulative return for all stocks. graph.
    cumul_returns = (avg_daily_returns + 1).cumprod() - 1

    return daily_returns, avg_daily_returns, cumul_returns


def graph_returns(df):
    sns.set(style="whitegrid")
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df)

    # plt.title('Daily Returns of Stocks')
    # plt.xlabel('Date')
    # plt.ylabel('Daily Return')
    # plt.save(f'label:{TODAY_DATE}.png')


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
def group_returns(returns, sector_dict):

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


def main():
    # FAULTY TICKERS
    #     HEXA-B,2023-11-30,Industrials

    stock_info_df = pd.read_csv("pitched_stock_info.csv")
    stock_info_df.columns = ['Stock symbol', 'Pitch date', 'Sector']

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

    # stratifies returns by sector. you can select any of the 3 dataframes
    grouped_returns = group_returns(cumul_returns, sector_dict)
    for returns in grouped_returns:
        print(f'SECTOR: {returns[0]}')
        print(returns[1].tail())

    # print(daily_returns.tail())
    # print(avg_daily_returns.tail())


if __name__ == '__main__':
    main()
