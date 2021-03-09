# Import libraries
import pandas as pd
import numpy as np
import logging
import yfinance as yf
from bs4 import BeautifulSoup
import os, time, shutil, glob, smtplib,ssl
import matplotlib.pyplot as plt
from urllib.request import urlopen, Request
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#logging details
def init_logging(config):
    format = '%(asctime)s %(process)d %(module)s %(levelname)s %(message)s'
logging.basicConfig(handlers=[logging.FileHandler('Stock_Analysis_Directory/Stock_News_Scraping_log.txt','w', 'utf-8')], level=logging.INFO,
                    format='%(asctime)s %(name)s %(message)s',datefmt='%a, %d %b %Y %H:%M:%S')
logging.getLogger('requests').setLevel(logging.CRITICAL)

my_file = os.path.join("Stock_Analysis_Directory")
logging.info("\n<-----------------------------------------------------------------OUTPUT_LOG_FILE-------------------------------------------------------------------------------------------------->")
logging.info("\n------------Creating directory Data_Analysis_Plots_Directory where all generated plots will be saved------------\n")
if not os.path.exists(my_file): #This method returns a Boolean value of class bool. This method returns True if path exists otherwise returns False.
        os.mkdir(my_file) #Returns error if the directory does not exists.


# Parameters
n = 3  # the # of article headlines displayed per ticker
tickers = ['TXMD', 'TSLA', 'AMZN']

# Get Data
finwiz_url = 'https://finviz.com/quote.ashx?t='
news_tables = {}

for ticker in tickers:
    url = finwiz_url + ticker
    req = Request(url=url, headers={'user-agent': 'my-app/0.0.1'})
    resp = urlopen(req)
    html = BeautifulSoup(resp, features="lxml")
    news_table = html.find(id='news-table')
    news_tables[ticker] = news_table

try:
    for ticker in tickers:
        df = news_tables[ticker]
        df_tr = df.findAll('tr')

        print('\n')
        print('Recent News Headlines for {}: '.format(ticker))

        for i, table_row in enumerate(df_tr):
            a_text = table_row.a.text
            td_text = table_row.td.text
            td_text = td_text.strip()
            print(a_text, '(', td_text, ')')
            if i == n - 1:
                break
except KeyError:
    pass

# Iterate through the news
parsed_news = []
for file_name, news_table in news_tables.items():
    for x in news_table.findAll('tr'):
        text = x.a.get_text()
        date_scrape = x.td.text.split()

        if len(date_scrape) == 1:
            time = date_scrape[0]

        else:
            date = date_scrape[0]
            time = date_scrape[1]

        ticker = file_name.split('_')[0]

        parsed_news.append([ticker, date, time, text])

# Sentiment Analysis
analyzer = SentimentIntensityAnalyzer()

columns = ['Ticker', 'Date', 'Time', 'Headline']
news = pd.DataFrame(parsed_news, columns=columns)
scores = news['Headline'].apply(analyzer.polarity_scores).tolist()

df_scores = pd.DataFrame(scores)
news = news.join(df_scores, rsuffix='_right')

# View Data
news['Date'] = pd.to_datetime(news.Date).dt.date

unique_ticker = news['Ticker'].unique().tolist()
news_dict = {name: news.loc[news['Ticker'] == name] for name in unique_ticker}

values = []
for ticker in tickers:
    dataframe = news_dict[ticker]
    dataframe = dataframe.set_index('Ticker')
    dataframe = dataframe.drop(columns=['Headline'])
    print('\n')
    print(dataframe.head())

    mean = round(dataframe['compound'].mean(), 2)
    values.append(mean)

df = pd.DataFrame(list(zip(tickers, values)), columns=['Ticker', 'Mean Sentiment'])
df = df.set_index('Ticker')
df = df.sort_values('Mean Sentiment', ascending=False)
print('\n')
print(df)
logging.info(df)

# #Emailing the result
# Analysis = pd.read_csv("<Your Path>\\Daily_Stock_Report\\OBV_Ranked.csv")  # Read in the ranked stocks
# top10 = Analysis.head(10)  # I want to see the 10 stocks in my analysis with the highest OBV values
# bottom10 = Analysis.tail(10)  # I also want to see the 10 stocks in my analysis with the lowest OBV values
# # This is where we write the body of our email. Add the top 10 and bottom 10 dataframes to include the results of your analysis
# Body_of_Email = """\
# Subject: Daily Stock Report
#
# Your highest ranked OBV stocks of the day:
#
# """ + top10.to_string(index=False) + """\
#
#
# Your lowest ranked OBV stocks of the day:
#
# """ + bottom10.to_string(index=False) + """\
#
#
# Sincerely,
# Your Computer"""
# context = ssl.create_default_context()
# Email_Port = 465  # If you are not using a gmail account, you will need to look up the port for your specific email host
# with smtplib.SMTP_SSL("smtp.gmail.com", Email_Port, context=context) as server:
#     server.login("<Your email>", "<Your email password>")  #  This statement is of the form: server.login(<Your email>, "Your email password")
#     server.sendmail("<Your email>", "<Email receiving message>", Body_of_Email)  # This statement is of the form: server.sendmail(<Your email>, <Email receiving message>, Body_of_Email)