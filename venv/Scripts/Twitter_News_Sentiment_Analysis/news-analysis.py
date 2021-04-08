import pandas as pd
import numpy as np 
import seaborn as sns
import tweepy as api
import tweepy as api
import flask as api
names = ["washingtonpost", "ABC", "SkyNews", "CNN", "BBCWorld", "nytimes", "NBCNews", "VICENews"]

i = 0

for name in names:
    tweets = api.user_timeline(screen_name =names[i], count = 200)
    df = tweet_analyzer.tweets_to_data_frame(tweets)
    df2 = df2.append(df)
    df2.sort_values("sentiment", inplace= True, ascending = False)
    i=i+1

def tweets_to_data_frame(self, tweets):
    df = pd.DataFrame()
    df1 = pd.DataFrame(data=[tweet.text for tweet in tweets], columns = ['tweets'])


    df1['ids']  = np.array([tweet.id for tweet in tweets])
    df1['len'] = np.array([len(tweet.text) for tweet in tweets])
    df1['date'] = np.array( [tweet.created_at for tweet in tweets])
    df1['likes'] = np.array([tweet.favourite_count for tweet in tweets])
    df1['retweets'] = np.array([tweet.retweet_count for tweet in tweets])
    df1['name'] = np.array([tweet.user.name for tweet in tweets])
    df1['link'] = 'www.twitter.com/i/web/status/' + df1['ids'].astype(str)

    df = df.append(df1)

    return df
    print("succcess")

class TweetAnalyzer():
    """ Functionality for cleaning and analysing content from the tweets """

    def analyze_sentiment(self, tweet):
        analysis = TextBlob(self.clean_tweet(tweet))

        if analysis.sentiment.polarity > 0.6:
            return analysis.sentiment.polarity * 100
        else:
            0   
            
    
