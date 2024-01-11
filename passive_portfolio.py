import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from dateutil.relativedelta import relativedelta
import seaborn as sns
import logging

# setting up logging module for debugging purposes
logging.basicConfig(
    # filename='get data - yfinance.log',
    # levels in ascending order : DEBUG, INFO, WARNING, ERROR, CRITICAL
    # everything below a level will not be printed in terminal at runtime
    level=logging.WARNING,
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

-group returns by sector
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

def main():
    # Tickers not working with yfinance at the moment
    #     'HEXA-B': '2023-11-30',

    # date format as 'YYYY-MM-DD' for yfinance compatibility
    tracking_dict = {
        'DOLE': '2023-12-14', # consumer staple
        'AC.PA': '2023-12-14', # discretionary
        'GF.SW': '2023-12-14', # industrials
        'EXP': '2023-12-07', # materials
        'IDCC': '2023-12-07', # infotech
        'JSE.L': '2023-12-07', # energy
        'RMV.L': '2023-11-30', # comms. services
        'RMS.PA': '2023-11-30', # discretionary
        'AI.PA': '2023-11-23',  # materials
        'IBE.MC': '2023-11-23', # energy
        'INGR': '2023-11-23', # consumer staples
        'GOOGL': '2023-11-16', # comms. services
        'CDNS': '2023-11-16', # infotech
        'P911.DE': '2023-11-09', # discretionary
        'GNC.L': '2023-11-09', # consumer staples
        'CAT': '2023-11-09', # industrials
    }

    sector_dict = {
        'DOLE': 'consumer staples',
        'AC.PA': 'discretionary',
        'GF.SW': 'industrials',
        'EXP': 'materials',
        'IDCC': 'infotech',
        'JSE.L': 'energy',
        'RMV.L': 'comms. services',
        'RMS.PA': 'discretionary',
        'AI.PA': ' materials',
        'IBE.MC': 'energy',
        'INGR': 'consumer staples',
        'GOOGL': 'comms. services',
        'CDNS': 'infotech',
        'P911.DE': 'discretionary',
        'GNC.L': 'consumer staples',
        'CAT': 'industrials'
    }

    # iterating from oldest to most recent pitch
    dates = sorted(list(set(tracking_dict.values())))
    logging.info(f'Dates: {dates}')

    # returns dataframe of historical prices for pitched stocks
    portfolio_history = fetch_pitched_stock_data(tracking_dict, dates)
    print(portfolio_history.tail())

    daily_returns, avg_daily_returns, cumul_returns = get_returns(portfolio_history, tracking_dict)

    print(daily_returns.tail())
    print(avg_daily_returns.tail())
    print(cumul_returns.tail())


if __name__ == '__main__':
    main()
