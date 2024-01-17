import random
import pandas as pd
import yfinance as yf
import openbb as bb
from datetime import datetime
from dateutil.relativedelta import relativedelta
# views = []
# for i in range(16):
#     views.append(random.randint(0,99)/100)

# note: for two adjacent dates, yf.download will fetch the adjacent close of the earlier date
# caps_df = yf.download('CAT', start='2023-11-09', end='2023-11-10')
# print(caps_df.tail())

# output = bb.obb.equity.price.historical("AAPL")
# df = output.to_dataframe()
# print(df.columns())


adj_date = datetime.strptime('2024-12-02','%Y-%m-%d')
print(adj_date.strftime('%Y-%m-%d'))