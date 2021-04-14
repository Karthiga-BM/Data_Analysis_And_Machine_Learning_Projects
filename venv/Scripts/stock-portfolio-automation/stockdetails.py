# Library imports
import datetime
import os
import shutil
# pip3 install smtplib
import smtplib
# pip3 install ssl
import ssl
import time

# pip3 install pandas
import pandas as pd
# pip3 install yfinance
import yfinance as yf
# pip3 install python-dateutil
from dateutil.relativedelta import relativedelta

# deletes the folder where the stocks data are stored
shutil.rmtree(
    "<path to this project>/Stocks")
# makes the folder where the stocks data are stored
os.mkdir("<path to this project>/Stocks")

# reads the excel file you created representing your portfolio
df = pd.read_excel(
    "<path to this project>/Long_Term_Portfolio.xlsx")
# converts the stocks and their weights to a pandas dataframe
tickers = df.to_dict()

# converts the date from a TimeStamp format to a string so that it is usable
j = 0
for k in tickers['Purchase Date']:
    tickers['Purchase Date'][j] = tickers['Purchase Date'][j].strftime(
        '%Y-%m-%d')
    j += 1

# number of stocks not downloaded
stocks_not_donwloaded = 0
# used for iteration
i = 0

# loop used for iteration through each ticker to collect the data
while (i < len(tickers["Ticker"])):
    try:
        # stores each ticker in your portfolio one by one as we go through the loop
        stock = tickers["Ticker"][i]
        temp = yf.Ticker(str(stock))
        # gets the historical data for the specific ticker
        data = temp.history(period="max")
        # stores the data to our folder in .csv format
        data.to_csv(
            "<path to this project>/Stocks/Stocks"+stock+".csv")
        # slows the script down a bit in order to not make very fast requests to the yahoo API
        time.sleep(1)
        i += 1
    except ValueError:
        # if there is an error downloading data print the name of that ticker
        print("Error with Downloading Data for " + stock)
        i = +1
print("Stocks Successfully Imported" + str(i - stocks_not_donwloaded))

cks.py
list_files = []  # array to store the stock data files

# stores each directory in the 'list_files' array
ticker_list = list(tickers['Ticker'].values())
for ticker in ticker_list:
    list_files.append("<path to this project>/Stocks/Stocks" + ticker + ".csv")

# stores the weights of each stock respectively in a list
weight_list = list(tickers['Weight'].values())
weight_sum = sum(weight_list)  # sum of the weights

# the rest of the weights left is added to the portfolio as cash
cash_weight = 1 - weight_sum
tickers['Ticker'][len(ticker_list)] = 'Nothing'
tickers['Weight'][len(ticker_list)] = cash_weight
tickers['Purchase Date'][len(ticker_list)] = str(
    datetime.datetime.now().date() - relativedelta(days=1))

totalRet = {}  # stores the total return of a stock since purchased in a dictionary
todaysRet = {}  # stores today's return of a stock in a dictionary
interval = 0  # used for iteration

# loops through the 'list_files' and makes calculations
while interval < len(list_files):
    Data = pd.read_csv(list_files[interval])  # reads the csv file of the specific stock

    boughtPrice = Data.loc[
        Data['Date'] == tickers['Purchase Date'][interval]]  # stores the stock's price on the day it was purchased
    currentPrice = Data.iloc[[-1]]  # stores the stock's last price
    dayBeforePrice = Data.iloc[[-2]]  # stores the stock's price 2 weekdays ago

    totalReturn = (currentPrice['Close'].values[0] / boughtPrice['Close'].values[
        0])  # calculates the stock's total return since purchases by dividing the current price by the purchase price
    totalRet[interval] = totalReturn  # adds the 'totalReturn' to the dictionary we created for them

    todaysReturn = (currentPrice['Close'].values[0] / dayBeforePrice['Close'].values[
        0])  # calculates the stock's return today by dividing the current price yesterday
    todaysRet[interval] = todaysReturn  # adds the 'todaysReturn' to the dictionary we created for them

    interval += 1

# adds 1 to the last row of 'totalRet' and 'todaysRet' as it is the return rate of the cash
totalRet[interval] = 1
todaysRet[interval] = 1

# add the new dictionaries we created to the old dictionary we stored the data in
tickers['totalReturn'] = totalRet
tickers['todaysReturn'] = todaysRet

# converts the dictionary 'tickers' to a pandas dataframe
df = pd.DataFrame.from_dict(tickers)

# iterating through each of the tickers and multipling its total return by its weight to get its total weighted return in the portfolio. after that we add them all up to find the true return based on weights.
sub = 0
totalWeightedReturn = 0
for i in df.index:
    sub = df.iloc[i]['Weight'] * df.iloc[i]['totalReturn']
    totalWeightedReturn += sub

# iterates through each of the tickers and multipling todays return by its weight to get today's weighted return in the portfolio. after that we add them all up to find the true return based on weights.
sub = 0
todaysWeightedReturn = 0
for t in df.index:
    sub = df.iloc[t]['Weight'] * df.iloc[t]['todaysReturn']
    todaysWeightedReturn += sub

# finds the top five movers in your portfolio
sub_df = df # duplicates your dataframe so that it does not change the original one
sub_df = sub_df[:-1] # deletes the last row in the dataframe as we don't want 'nothing' to be included in the calculations
top_five_movers = sub_df.sort_values(by=['todaysReturn']).tail(5) # sorts the rows by smallest to greatest of 'todaysReturn' then gets the last five rows
top_five_movers = top_five_movers['Ticker'].values[:] # stores the tickers only in the 5 rows

# body of the email that will be sent
email = """\
Subject: Automated Stock Analysis
Your Portfolio:
""" + df.to_string(index=True) + """\
All Time Return:
""" + str(totalWeightedReturn) + """\
Today's Return:
""" + str(todaysWeightedReturn) + """\
Top 5 Movers Today:
""" + str(top_five_movers) + """\
"""
cont = ssl.create_default_context() # each connection should have its own context, this creates the new context for every connection we are doing.
port = 465 # smtp port for your email host. this is the gmail host. you will need to look up the post for your email host.
with smtplib.SMTP_SSL("smtp.gmail.com", port, context=cont) as server:
    server.login("<sender email>", "<login password>") # logs in to the sender email
    server.sendmail("<sender email>", "<receiver email>", email) # sends an email with
