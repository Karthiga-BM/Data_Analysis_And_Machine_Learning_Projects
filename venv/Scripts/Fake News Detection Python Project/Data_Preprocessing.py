import pandas as pd
import seaborn as sns
import sklearn.model_selection as ms
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer  # bag of words representation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from collections import Counter
import string
import nltk
import json
import re
import urllib
from nltk.tokenize import word_tokenize #this tokenize the texts.That means basically split into words.
from nltk.corpus import wordnet
from nltk.corpus import stopwords
import logging
import os
from nltk import FreqDist
from nltk.util import ngrams
import matplotlib.pyplot as mpl
import logging
import pickle
import statistics
from collections import Counter
from plotly.graph_objs import *
import plotly.graph_objects as go
from nltk.stem import WordNetLemmatizer
import textblob
from textblob import TextBlob
import unicodedata
import coloredlogs,logging
import wordcloud
from wordcloud import WordCloud
import sklearn.model_selection as ms
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import WordNetLemmatizer

ADDITIONAL_STOPWORDS = ['covfefe']
stop_words = set(stopwords.words('english'))
mpl.rc('figure', max_open_warning = 0)

#plt.rcParams.update({'figure.max_open_warning': 0})
#---------------------------------------------------------------------------------------COLOURED LOG----------------------------------------------------------------------
BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(8)

RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"
BOLD_SEQ = "\033[1m"
def formatter_message(message, use_color = True):
    if use_color:
        message = message.replace("$RESET", RESET_SEQ).replace("$BOLD", BOLD_SEQ)
    else:
        message = message.replace("$RESET", "").replace("$BOLD", "")
    return message

COLORS = {
    'WARNING': WHITE,
    'INFO': BLUE,
    'DEBUG': BLUE,
    'CRITICAL': YELLOW,
    'ERROR': RED
}
class ColoredFormatter(logging.Formatter):
    def __init__(self, msg, use_color = True):
        logging.Formatter.__init__(self, msg)
        self.use_color = use_color

    def format(self, record):
        levelname = record.levelname
        if self.use_color and levelname in COLORS:
            levelname_color = COLOR_SEQ % (30 + COLORS[levelname]) + levelname + RESET_SEQ
            record.levelname = levelname_color
        return logging.Formatter.format(self, record)
# Custom logger class with multiple destinations
class ColoredLogger(logging.Logger):
    FORMAT = "[$BOLD%(name)-20s$RESET][%(levelname)-18s]  %(message)s ($BOLD%(filename)s$RESET:%(lineno)d)"
    COLOR_FORMAT = formatter_message(FORMAT, True)
    def __init__(self, name):
        logging.Logger.__init__(self, name, logging.DEBUG)

        color_formatter = ColoredFormatter(self.COLOR_FORMAT)

        console = logging.StreamHandler()
        console.setFormatter(color_formatter)

        self.addHandler(console)
        return
logging.setLoggerClass(ColoredLogger)
#----------------------------------------------------------------------------COLORED LOG-------------------------------------------------------
#<---------------------------------------------------------------------------------------------------------------------------------------------------------------->
#<---------------------------------------------------------------------DATA PREPROCESSING---------------------------------------------------------------------->
#<---------------------------------------------------------------------------------------------------------------------------------------------------------------->
#<-----------------------------------Creating a log file to save the output------------------------------------------------------------------>
#This file stores information and output on data preprocessing step
def init_logging(config):
    format = '%(asctime)s %(process)d %(module)s %(levelname)s %(message)s'
logging.basicConfig(handlers=[logging.FileHandler('Dataset_Preprocessing_log.txt','w', 'utf-8')], level=logging.INFO,
                    format='%(asctime)s %(name)s %(message)s',datefmt='%a, %d %b %Y %H:%M:%S')
logging.getLogger('requests').setLevel(logging.CRITICAL)
#coloredlogs.install(level='DEBUG')

#Creating a directory where all plots will be saved
#I created a directory named "Data_Analysis_Plots_Directory" to store all the generated plots.
#Reference:https://stackoverflow.com/questions/11373610/save-matplotlib-file-to-a-directory
my_file = os.path.join("Data_Analysis_Plots_Directory")
logging.info("\n<-----------------------------------------------------------------OUTPUT_LOG_FILE-------------------------------------------------------------------------------------------------->")
logging.info("\n------------Creating directory Data_Analysis_Plots_Directory where all generated plots will be saved------------\n")
if not os.path.exists(my_file): #This method returns a Boolean value of class bool. This method returns True if path exists otherwise returns False.
        os.mkdir(my_file) #Returns error if the directory does not exists.


#Loading the json file into the pandas dataframe
def load_data(input_json_file, class_column_label):

    df = pd.read_json(input_json_file, lines=True)  # load data into a pandas Data Frame

    logging.info("\n--------------------------------------------DATASET INFORMATION -------------------------------------------------")

    logging.info("\n-----------------------------------------GETTING THE DIMENSIONALITY OF THE DATAFRAME----------------------------")
    logging.info(df.shape)#Return a tuple representing the dimensionality of the DataFrame.

    logging.info("\n----------------------------------------GETTING THE INDEX VALUE IF THE DATAFRAME-------------------------------------")
    logging.info(df.index)#Pandas DataFrame index and columns attributes allow us to get the rows and columns label values.

    logging.info(df.info(verbose=None, buf=None, max_cols=None, memory_usage=None, null_counts=None))
    print("------------------------------------------------The FIRST N ROWS--------------------------------------------------\n", df.head())
    print("------------------------------------------------The DIMENSIONALITY OF DATAFRAME-----------------------------------------------------\n", df.shape)
    print("-------------------------------------The Total number of elements in the Dataframe------------------------\n",df.size)
    print("-------------------------------------The  number of dimensions of the Dataframe------------------------\n",df.ndim)
    print("--------Some Calculation------------",df.describe())
    



# Getting the labels from the dataframe
    print("\n--------------------------------------------------- Creating class labels in the Dataset_Preprocessing_log_file--------------------------------------------------------------")

#Distribution of classes for prediction
    logging.info("\n--------------------------DISTRIBUTION OF CLASSES FOR PREDICTION--------------------------------")
    logging.info("\n----------------------------------------CLASS LABELS & EXAMPLE DISTRIBUTION-----------------------------------------------------")
    logging.info(df.groupby(class_column_label).size())



# Plotting class frequency and save plot:
    logging.info("\n----------------------------------------------CLASS FREQUENCY PLOT------------------------------------------------------------------")
    class_frequency = sns.countplot(x="is_sarcastic", palette="Paired", data=df)
    mpl.title('Class Frequency Plot', fontsize=14)
    mpl.show()
    class_frequency.figure.savefig("Data_Analysis_Plots_Directory\class_frequency.png")
    logging.info("The Class frequency plot has been successfully created and saved in Data_Analysis_Plots_Directory\Class_Frequency_Plot.png")
    return df


def remove_punctuation_and_lowercase(headline):
    clean_headline = ''.join([s for s in headline if s not in string.punctuation])
    return clean_headline.lower()


def clean_text(df, text_field_name):
    df[text_field_name] = df[text_field_name].apply(remove_punctuation_and_lowercase)
    return df

# #--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------Creating a train test Split---------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def create_train_and_test_data(data_frame ):
    train_data, test_data = train_test_split(data_frame, test_size= 0.25, random_state=40)
    logging.info("\n------------------------------------------CREATING TRAIN & TEST SPLITS------------------------------------")
    logging.info(
        "\n------------------------------------------Class Frequency Of Train Data-----------------------------------------------------------")
    logging.info(train_data.groupby("is_sarcastic").size())
    logging.info("\n---------------------------------Class FRequency Of Test Data-----------------------------------")
    logging.info(test_data.groupby("is_sarcastic").size())
    logging.info("\n----------------------------------Plotting class frequency of train data-------------------------------------")
    train_news = sns.countplot(x='is_sarcastic', data=train_data, palette='Paired')
    mpl.title('Train Data Frequency Plot', fontsize=14)
    mpl.show()
    train_news.figure.savefig("Data_Analysis_Plots_Directory\Train_Data frequency Plot.png")
    logging.info(
        "\n The Class frequency plot of train data has been successfully created and saved in Data_Analysis_Plots_Directory\Train Data frequency plot.png")
    logging.info(
        "\n-----------------------------Plotting class frequency of test data----------------------------------------")
    test_news = sns.countplot(x='is_sarcastic', data=test_data, palette='Paired')
    mpl.title('Test Data Frequency Plot', fontsize=14)
    mpl.show()
    test_news.figure.savefig("Data_Analysis_Plots_Directory\Test Data frequency Plot .png")
    logging.info(
        "\n The class frequency plot of test data has been successfully created and saved in Data_Analysis_Plots_Directory\Test Data frequency plot.png")
    logging.info("\n--------------------------------------------------------Shape of the train and test data------------------------------------------")
    logging.info(train_data.shape)
    logging.info(test_data.shape)

    print("\n---------------------------The shape of train and test data split--------------------------------------------------")
    print(train_data.shape)
    print(test_data.shape)

#Quality check of dataset
    logging.info("\n-------------------------Checking the dataset quality---------------------")
    train_data.isnull().sum()
    train_data.info()
    logging.info("\n Check finished..")
    test_data.isnull().sum()
    test_data.info()

#Getting the labels from the train and tests splits
    logging.info("------------------------------------------------Labels from the Train and Tests Splits------------------------------------------")
    Y_train = train_data['is_sarcastic'].values
    Y_test = test_data['is_sarcastic'].values
    logging.info(Y_train)
    logging.info(Y_test)


# Saving the train and test split to json files
    logging.info("--------------------------------------Saving the train and test splits in desired format-------------------------------------")
    train_data.to_json(r'train.json')
    logging.info("\n The train data is saved as train.json")
    logging.info("\n The test data is saved as test.json")
    test_data.to_json(r'test.json')
    return train_data, test_data


#<---------------------------------------------------------------------------------------------------------------------------------------------------------------->
#<----------------------------------------------------------------- Exploratory Data Analysis--------------------------------------------------------------------->
#<---------------------------------------------------------------------------------------------------------------------------------------------------------------->


def create_data_analysis_report(data_frame):


    real_news, fake_news = [news for _, news in data_frame.groupby(data_frame['is_sarcastic'] == 1)]
    logging.info("---------------------------------Shape of Real and Fake news in training data----------------------------------------------------------------------")
    logging.info(real_news.shape)
    logging.info(fake_news.shape)
    print("\n-------------------------------------------Shape of Real and Fake News in training data------------------------------------------------------------------------")
    print("\n Real News:",real_news.shape)
    print("\n Fake News:",fake_news.shape)



#<-------------------------------------------------EXPLORING ARTICLE HEADLINE TEXT---------------------------------------------------------------------------------------------->
    words_per_headline_plot_t = real_news["headline"].apply(lambda x: len(x.split()))
    stdev_t_head = statistics.stdev(words_per_headline_plot_t)
    words_per_headline_t = words_per_headline_plot_t.sum() / len(real_news["headline"])

    words_per_headline_plot_f = fake_news["headline"].apply(lambda x: len(x.split()))
    stdev_f_head = statistics.stdev(words_per_headline_plot_f)
    words_per_headline_f = words_per_headline_plot_f.sum() / len(fake_news["headline"])

    logging.info("\n--------------------------------------------------------Exploring Article Headline Text-----------------------------------------------------------")
    logging.info("\n-------------------------------------------------Average Number and Standard Deviation----------------------------------------")
    logging.info("\nThe average number of words in a real news Headline is :")
    logging.info(words_per_headline_t)
    logging.info("\nThe average number of words in a fake news Headline is :")
    logging.info(words_per_headline_f)
    logging.info("\nThe standard deviation in real news article lengths is:")
    logging.info(stdev_t_head)
    logging.info("\nThe standard deviation in fake news article lengths is :")
    logging.info(stdev_f_head)
    print("-------------------------------------------Averge number and Standard Deviation------------------------------------------------")
    print("The average number of words in a real news headline is ", words_per_headline_t)
    print("The average number of words in a fake news headline is ", words_per_headline_f)

    print("The standard deviation in real news articles' headline lengths is ", stdev_t_head)
    print("The standard deviation in fake news articles' headline lengths is ", stdev_f_head)


#Plotting the average and standard deviation diagram
    fig, ax = mpl.subplots(1, 2, figsize=(10, 6))
    words_per_headline_plot = sns.distplot(words_per_headline_plot_t, ax=ax[0], color="darkblue", rug=True).set_title(
        "Number of Words in Real News Headline")
    words_per_headline_plot = sns.distplot(words_per_headline_plot_f, ax=ax[1], color="red", rug=True).set_title(
        "Number of Words in Fake News Headline")
    mpl.show()
    words_per_headline_plot.figure.savefig("Data_Analysis_Plots_Directory\words_per_headline_plot.png")

#--------------------------------------------ARTICLE HEADLINE SENTIMENT ANALYSIS-----------------------------------------------------
    headline_polarity_true = pd.DataFrame(columns=["Headline", "sentiment"])
    for headline in real_news["headline"]:
        headline = TextBlob(headline)
        headline_polarity_true = headline_polarity_true.append(
            pd.Series([headline, headline.sentiment.polarity], index=headline_polarity_true.columns), ignore_index=True)


    headline_polarity_fake = pd.DataFrame(columns=["Headline", "sentiment"])
    for headline in fake_news["headline"]:
        headline = TextBlob(headline)
        headline_polarity_fake = headline_polarity_fake.append(
            pd.Series([headline, headline.sentiment.polarity], index=headline_polarity_fake.columns), ignore_index=True)

    headline_polarity_true_sm = statistics.mean(headline_polarity_true["sentiment"])
    headline_polarity_fake_sm = statistics.mean(headline_polarity_fake["sentiment"])


    logging.info(
        "\n-------------------------------------------Sentiment Analysis of Article Headline Text-----------------------------------------------------------")
    logging.info("\nThe headline sentiment analysis result for real_news :")
    logging.info(headline_polarity_true_sm)
    logging.info("\nThe headline sentiment analysis result for real_news : :")
    logging.info(headline_polarity_fake_sm)
    logging.info(
        "\nPlotting headline_sentiment_plot and saved at Data_Analysis_Plots_Directory\headline_sentiment_analysis_plot:")

    fig, ax = mpl.subplots(1, 2, figsize=(10, 6))
    headline_sentiment_plot = sns.distplot(headline_polarity_true["sentiment"], ax=ax[0], color="darkblue",
                                           rug=True).set_title("Real News Headline Sentiments")
    headline_sentiment_plot = sns.distplot(headline_polarity_fake["sentiment"], ax=ax[1], color="red", rug=True).set_title(
        "Fake News Headline Sentiments")
    mpl.show()
    headline_sentiment_plot.figure.savefig("Data_Analysis_Plots_Directory\headline_sentiment_analysis_plot.png")

# #---------------------------------------Computing bigrams in Real News headline-------------------------------------------------------------------
    lemmatizer = WordNetLemmatizer()


    words_in_real_news_headline = []  # all tokens in true articles
    words_in_fake_news_headline = []

    words_in_real_news_headline_with_no_stopwords = []  # all tokens in true articles
    words_in_fake_news_headline_with_no_stopwords = []  # all tokens in fake articles



#--------------------------------------------------Processinng ngram--------------------------------------------------------------------------------
    process(real_news, words_in_real_news_headline)
    process(fake_news, words_in_fake_news_headline)

    bigrams_real_news_headline = zip(words_in_real_news_headline, words_in_real_news_headline[1:])
    bigram_counts_real_news_headline = Counter(bigrams_real_news_headline)
    df = pd.DataFrame(bigram_counts_real_news_headline.most_common(20), columns=["Bigram_Real_News", "Frequency"])
    bigrams_real_news_headline = df
    logging.info(bigrams_real_news_headline)


    df.sort_values(by='Frequency', ascending=False)

    df.plot.barh(x='Bigram_Real_News', y='Frequency', title="Top Bigrams in Real News Headline").invert_yaxis()
    mpl.savefig("Data_Analysis_Plots_Directory\ real_news_top_Bigrams_plot.png", bbox_inches = "tight")
    mpl.show()






#---------------------------------------Computing bigrams in Fake News headline-------------------------------------------------------------------

    bigrams_fake_news_headline = zip(words_in_fake_news_headline, words_in_fake_news_headline[1:])
    bigram_counts_fake_news_headline = Counter(bigrams_fake_news_headline)
    df = pd.DataFrame(bigram_counts_fake_news_headline.most_common(20), columns=["Bigram_Fake_News", "Frequency"])
    bigrams_fake_news_headline = df
    logging.info(bigrams_fake_news_headline)


    df.sort_values(by='Frequency', ascending=False)

    df.plot.barh(x='Bigram_Fake_News', y='Frequency', title="Top Bigram in Fake News Headline").invert_yaxis()
    mpl.savefig("Data_Analysis_Plots_Directory\ fake_news_top_Bigrams_plot.png", bbox_inches="tight")
    mpl.show()



#---------------------------------------Computing trigrams in real news headline-------------------------------------------------------------------

    trigrams_real_news_headline = zip(words_in_real_news_headline, words_in_real_news_headline[1:], words_in_real_news_headline[2:])
    trigram_counts_real_news_headline = Counter(trigrams_real_news_headline)
    df = pd.DataFrame(trigram_counts_real_news_headline.most_common(20), columns=["Trigram_Real_News", "Frequency"])
    trigrams_real_news_headline = df
    logging.info(trigrams_real_news_headline)


    df.sort_values(by='Frequency', ascending=False)

    df.plot.barh(x='Trigram_Real_News', y='Frequency', title="Top Tigrams in Real News Headline").invert_yaxis()
    mpl.savefig("Data_Analysis_Plots_Directory\ real_news_top_Trigrams_plot.png", bbox_inches="tight")
    mpl.show()

#---------------------------------------Computing trigrams in fake news headline-------------------------------------------------------------------
    trigrams_fake_news_headline = zip(words_in_fake_news_headline, words_in_fake_news_headline[1:], words_in_fake_news_headline[2:])
    trigram_counts_fake_news_headline = Counter(trigrams_fake_news_headline)
    df = pd.DataFrame(trigram_counts_fake_news_headline.most_common(20), columns=["Trigram_Fake_News", "Frequency"])
    trigrams_fake_news_headline = df
    logging.info(trigrams_fake_news_headline)


    df.sort_values(by='Frequency', ascending=False)

    df.plot.barh(x='Trigram_Fake_News', y='Frequency', title="Top Trigrams in Fake News Headline").invert_yaxis()
    mpl.savefig("Data_Analysis_Plots_Directory\ fake_news_top_Trigrams_plot.png", bbox_inches="tight")
    mpl.show()

#---------------------------------------Computing unigram in real news headline-------------------------------------------------------------------

    wordcounts_r = Counter(words_in_real_news_headline)
    mostcommon_r = Counter(wordcounts_r).most_common(20)
    df = pd.DataFrame(mostcommon_r, columns=["Unigram_Real_News", "Frequency"])
    logging.info(df)


    df.sort_values(by='Frequency', ascending=False)

    df.plot.barh(x='Unigram_Real_News', y='Frequency', title="Top Unigram in Real News Headline").invert_yaxis()
    mpl.savefig("Data_Analysis_Plots_Directory\ real_news_top_unigrams_plot.png", bbox_inches="tight")
    mpl.show()

    r_plot = dict(mostcommon_r)
    mostcommon_r = df.reset_index(drop=True)
    mostcommon_r = df['Unigram_Real_News'].tolist()

    r_wc = WordCloud(max_words=25,relative_scaling=1,background_color ='white', normalize_plurals=False).generate_from_frequencies(r_plot)

    mpl.imshow(r_wc)
    mpl.title("Plot of Most Frequent Words in Real News")
    mpl.savefig("Data_Analysis_Plots_Directory\ real_news_top_unigrams_wc_plot.png", bbox_inches="tight")
    mpl.show()

#---------------------------------------Computing unigram in fake news headline-------------------------------------------------------------------



    wordcounts_f = Counter(words_in_fake_news_headline)
    mostcommon_f = Counter(wordcounts_f).most_common(20)
    df = pd.DataFrame(mostcommon_f, columns=["Unigram_Fake_News", "Frequency"])
    logging.info(df)


    df.sort_values(by='Frequency', ascending=False)

    df.plot.barh(x='Unigram_Fake_News', y='Frequency', title="Top Unigram in Fake News Headline").invert_yaxis()
    mpl.savefig("Data_Analysis_Plots_Directory\ fake_news_top_unigrams_plot.png", bbox_inches="tight")
    mpl.show()

    f_plot = dict(mostcommon_f)
    mostcommon_f = df.reset_index(drop=True)
    mostcommon_f = df['Unigram_Fake_News'].tolist()

    f_wc = WordCloud(max_words=25,relative_scaling=1,background_color ='white', normalize_plurals=False).generate_from_frequencies(f_plot)

    mpl.imshow(f_wc)
    mpl.title("Plot of Most Frequent Words in Fake News")
    mpl.savefig("Data_Analysis_Plots_Directory\ fake_news_top_unigrams_wc_plot.png", bbox_inches="tight")
    mpl.show()

#Of the top 20 words in each class, 9 words are common
    logging.info("-------------------------Of the top 20 words in each class, Number of words  that are common------------------------------------------")
    logging.info(len(set(mostcommon_r) & set(mostcommon_f)))
#--------------------------------------------------Processinng ngram with no stop words--------------------------------------------------------------------------------

    process_no_stopwords(real_news, words_in_real_news_headline_with_no_stopwords)
    process_no_stopwords(fake_news, words_in_fake_news_headline_with_no_stopwords)

#---------------------------------------Computing bigrams in Real News headline with no stop word-------------------------------------------------------------------

    bigrams_real_news_headline = zip(words_in_real_news_headline_with_no_stopwords, words_in_real_news_headline_with_no_stopwords[1:])
    bigram_counts_real_news_headline = Counter(bigrams_real_news_headline)
    df = pd.DataFrame(bigram_counts_real_news_headline.most_common(20), columns=["Bigram_Real_News_with_no_stopwords", "Frequency"])
    bigrams_real_news_headline = df
    logging.info(bigrams_real_news_headline)


    df.sort_values(by='Frequency', ascending=False)

    df.plot.barh(x='Bigram_Real_News_with_no_stopwords', y='Frequency', title="Top Bigrams in Real News Headline with no stop words").invert_yaxis()
    mpl.savefig("Data_Analysis_Plots_Directory\ real_news_top_Bigrams_with_no_stop_words_plot.png", bbox_inches = "tight")
    mpl.show()






#---------------------------------------Computing bigrams in Fake News headline with no stop word-------------------------------------------------------------------

    bigrams_fake_news_headline = zip(words_in_fake_news_headline_with_no_stopwords, words_in_fake_news_headline_with_no_stopwords[1:])
    bigram_counts_fake_news_headline = Counter(bigrams_fake_news_headline)
    df = pd.DataFrame(bigram_counts_fake_news_headline.most_common(20), columns=["Bigram_Fake_News_no_stopwords", "Frequency"])
    bigrams_fake_news_headline = df
    logging.info(bigrams_fake_news_headline)


    df.sort_values(by='Frequency', ascending=False)

    df.plot.barh(x='Bigram_Fake_News_no_stopwords', y='Frequency', title="Top Bigram in Fake News Headline with no stop words").invert_yaxis()
    mpl.savefig("Data_Analysis_Plots_Directory\ fake_news_top_Bigrams_no_stopwords_plot.png", bbox_inches="tight")
    mpl.show()



#---------------------------------------Computing trigrams in real news headline with no stop words-------------------------------------------------------------------

    trigrams_real_news_headline = zip(words_in_real_news_headline_with_no_stopwords, words_in_real_news_headline_with_no_stopwords[1:],words_in_real_news_headline_with_no_stopwords[2:])
    trigram_counts_real_news_headline = Counter(trigrams_real_news_headline)
    df = pd.DataFrame(trigram_counts_real_news_headline.most_common(20), columns=["Trigram_Real_News_no_stopwords", "Frequency"])
    trigrams_real_news_headline = df
    logging.info(trigrams_real_news_headline)
    df.sort_values(by='Frequency', ascending=False)
    df.plot.barh(x='Trigram_Real_News_no_stopwords', y='Frequency', title="Top Tigrams in Real News Headline with no stop words").invert_yaxis()
    mpl.savefig("Data_Analysis_Plots_Directory\ real_news_top_Trigrams_no_stopwords_plot.png", bbox_inches="tight")
    mpl.show()

#---------------------------------------Computing trigrams in fake news headline with no stop words-------------------------------------------------------------------
    trigrams_fake_news_headline = zip(words_in_fake_news_headline_with_no_stopwords, words_in_fake_news_headline_with_no_stopwords[1:], words_in_fake_news_headline_with_no_stopwords[2:])
    trigram_counts_fake_news_headline = Counter(trigrams_fake_news_headline)
    df = pd.DataFrame(trigram_counts_fake_news_headline.most_common(20), columns=["Trigram_Fake_News_no_stopwords", "Frequency"])
    trigrams_fake_news_headline = df
    logging.info(trigrams_fake_news_headline)


    df.sort_values(by='Frequency', ascending=False)

    df.plot.barh(x='Trigram_Fake_News_no_stopwords', y='Frequency', title="Top Trigrams in Fake News Headline with no stop words").invert_yaxis()
    mpl.savefig("Data_Analysis_Plots_Directory\ fake_news_top_Trigrams_no_stopwords_plot.png", bbox_inches="tight")
    mpl.show()

#---------------------------------------Computing unigram in real news headline with no stop words-------------------------------------------------------------------

    wordcounts_r = Counter(words_in_real_news_headline_with_no_stopwords)
    mostcommon_r = Counter(wordcounts_r).most_common(20)
    df = pd.DataFrame(mostcommon_r, columns=["Unigram_Real_News_no_stopwords", "Frequency"])
    logging.info(df)
    df.sort_values(by='Frequency', ascending=False)
    df.plot.barh(x='Unigram_Real_News_no_stopwords', y='Frequency', title="Top Unigram in Real News Headline with no stop words").invert_yaxis()
    mpl.savefig("Data_Analysis_Plots_Directory\ real_news_top_unigrams_no_stopwords_plot.png", bbox_inches="tight")
    mpl.show()

    r_plot = dict(mostcommon_r)
    mostcommon_r = df.reset_index(drop=True)
    mostcommon_r = df['Unigram_Real_News_no_stopwords'].tolist()

    r_wc = WordCloud(max_words=25, relative_scaling=1, background_color='white',
                     normalize_plurals=False).generate_from_frequencies(r_plot)

    mpl.imshow(r_wc)
    mpl.title("Plot of Most Frequent Words with no stop words in Real News")
    mpl.savefig("Data_Analysis_Plots_Directory\ real_news_top_unigrams_with_o_stopwords_wc_plot.png", bbox_inches="tight")
    mpl.show()

#---------------------------------------Computing unigram in fake news headline with stop words-------------------------------------------------------------------

    wordcounts_f = Counter(words_in_fake_news_headline_with_no_stopwords)
    mostcommon_f = Counter(wordcounts_f).most_common(20)
    df = pd.DataFrame(mostcommon_f, columns=["Unigram_Fake_News_no_stopwords", "Frequency"])
    logging.info(df)
    df.sort_values(by='Frequency', ascending=False)
    df.plot.barh(x='Unigram_Fake_News_no_stopwords', y='Frequency', title="Top Unigram in Fake News Headline no_stopwords").invert_yaxis()
    mpl.savefig("Data_Analysis_Plots_Directory\ fake_news_top_unigrams_no_stopwords_plot.png", bbox_inches="tight")
    mpl.show()

    f_plot = dict(mostcommon_f)
    mostcommon_f = df.reset_index(drop=True)
    mostcommon_f = df['Unigram_Fake_News_no_stopwords'].tolist()

    f_wc = WordCloud(max_words=25, relative_scaling=1, background_color='white',
                     normalize_plurals=False).generate_from_frequencies(f_plot)

    mpl.imshow(f_wc)
    mpl.title("Plot of Most Frequent Words wit no stop words in Fake News")
    mpl.savefig("Data_Analysis_Plots_Directory\ fake_news_top_unigrams_no_stopwords_wc_plot.png", bbox_inches="tight")
    mpl.show()
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def process(df, word_headline_list):


    for article in (df["headline"]):
        words = word_tokenize(article)

        for w in words:
            word_headline_list.append(w)

#---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def process_no_stopwords(df, word_headline_list):


    for article in (df["headline"]):
        words = word_tokenize(article)
        words = [word for word in words if
              word not in string.punctuation and word not in stop_words]  # punctuation, stopwords----this line should be added to calculate ngrams with no stop words
        for w in words:
            word_headline_list.append(w)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
def process_article(df, words_in_each_article):
    for article in (df["headline"]):
        words = word_tokenize(article)

        for w in words:
            words_in_each_article.append(w)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Data cleaning
def basic_clean(df, word_list):
    for article in (df["headline"]):
        words = word_tokenize(article)
        words = [word.lower() for word in words if word.isalpha()]  # lowercase
        words = [word for word in words if
                 word not in string.punctuation and word not in stop_words]  # punctuation, stopwords

        for w in words:
            word_list.append(w)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == "__main__":
    input_json = 'fake_news.json'
    df = load_data(input_json, 'is_sarcastic')
    df_clean_data = clean_text(df, "headline")
    training_data, testing_data = create_train_and_test_data(df_clean_data)
    create_data_analysis_report(training_data)





