import yfinance as yf
import pandas as pd
import numpy as np
from time import perf_counter
from datetime import datetime
import logging

# setting up logging module for debugging purposes
logging.basicConfig(
    # filename='get data - yfinance.log',
    # levels in ascending order : DEBUG, INFO, WARNING, ERROR, CRITICAL. everything below a level will not be printed in terminal at runtime
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)


def update_portfolio(df):
    # the content being updated - PRICES NOT INCLUDED
    names = []
    sectors = []
    countries = []
    currencies = []

    # grabbing array of symbols from portfolio
    symbols = df['Stock symbol']
    logging.info(yf.Ticker('AAPL').fast_info.keys())

    # parsing array as string for use by yfinance - ['GS', 'MCD', 'MMM'] --> 'GS MCD MMM'
    # symbols = ''.join(symbol + ' ' for symbol in symbols)

    # iterates through each stock symbol and fetches info, appending it to several lists
    for symbol in symbols:
        ticker = yf.Ticker(symbol)

        name = ticker.info['longName']
        sector = ticker.info['sector']
        country = ticker.info['country']
        currency = ticker.fast_info['currency']

        # print(ticker.info)
        logging.info(f'name: {name}, sector: {sector}, country: {country}, currency: {currency}')

        names.append(name)
        sectors.append(sector)
        countries.append(country)
        currencies.append(currency)

    # replaces relevant dataframe columns with updated information
    df['Company name'] = names
    df['Sector'] = sectors
    df['Country'] = countries
    df['Currency'] = currencies

    return df


def main():
    # PURPOSE: updates portfolio data (excluding prices - which will be done in the PnL script

    # NOTE: if yfinance doesn't work, it is likely because Yahoo Finance
    # updated their decryption key. They've encrypted some information and
    # the decryption keys change often, so make sure to update yfinance regularly

    # headers used in portfolio spreadsheet
    headers = ['Company name', 'Stock symbol', 'Currency', 'Current price', 'Buy/sell', 'Buy/sell price', 'Qty. shares', 'Buy/sell date', 'Target price', 'Stop loss', 'Country', 'Sector', 'Ghosal sector', 'Weight', 'Standardised current price', 'Standardised buy/sell price', 'P&L', 'Return']

    # date will be added to exported file
    today = datetime.today().strftime('%Y-%m-%d')

    # reading in portfolio spreadsheet as pandas dataframe
    df = pd.read_excel('portfolio.xlsx', index_col=False)

    # operating on portfolio dataframe
    df = update_portfolio(df)


    # writes out updated df to 2 Excel sheets
    # this one is for logging purposes
    df.to_excel(f'portfolio - {today}.xlsx', index=False)

    # timestamped so we don't need to constantly change the date when reading the current portfolio
    df.to_excel('portfolio.xlsx', index=False)


if __name__ == '__main__':
    main()
