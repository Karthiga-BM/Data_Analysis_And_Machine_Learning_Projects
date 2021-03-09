import pandas as pd
import logging
import numpy as np
import seaborn as sns
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn import metrics
from nltk.corpus import stopwords
from sklearn.metrics import mean_squared_error
import pickle
from nltk import classify
from nltk import NaiveBayesClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import matplotlib.pyplot as mpl

#---------------------------------------------------Creating a log file to save the output-----------------------------------------------------------------------------------
def init_logging(config):
    format = '%(asctime)s %(process)d %(module)s %(levelname)s %(message)s'
logging.basicConfig(handlers=[logging.FileHandler('Modelling_SVM_5 cross validation_Tfidf_log.txt','w', 'utf-8')], level=logging.INFO,
                    format='%(asctime)s %(name)s %(message)s',datefmt='%a, %d %b %Y %H:%M:%S')
logging.info(("\n---------------------------------------------------------Modelling_log--------------------------------------------------------------------------------------"))
logging.getLogger('requests').setLevel(logging.CRITICAL)

#-----------------------------------------------------Creating train text and train label values--------------------------------------------------------------------------------
def train_data_values(json_file):
    dataset =pd.read_json(json_file)
    text = list(dataset['headline'])
    labels = list(dataset['is_sarcastic'])

    if len(text) != len(labels):
        exit("[ERROR]: there is a different number of input text strings and output labels")
    logging.info("\n The train text and train labels are created successfully")
    return text, labels

def test_data_values(json_file):
    dataset =pd.read_json(json_file)
    text = list(dataset['headline'])
    labels = list(dataset['is_sarcastic'])

    if len(text) != len(labels):
        exit("[ERROR]: there is a different number of input text strings and output labels")
    logging.info("\n The test text and test labels are created successfully")
    return text, labels


def bag_of_words_representation(text2bow, headlines_list, test=False):

    if test:
        input_fts = text2bow.transform(headlines_list).toarray()
    else:
        input_fts = text2bow.fit_transform(headlines_list).toarray()
    logging.info(text2bow.get_feature_names())
    return input_fts


def tf_idf_representation(text2tfidf, headlines_list, test=False):

    if test:
        input_fts = text2tfidf.transform(headlines_list).toarray()
    else:
        input_fts = text2tfidf.fit_transform(headlines_list).toarray()

    return input_fts

if __name__ == "__main__":
    train_text, train_labels = train_data_values('train.json')
    test_text, test_labels = test_data_values('test.json')

    for number_of_features in [100,500,1000,5000, 8000]:
        logging.info("-----------------------------TRAINING SVM Models Tfidf MODELS WITH {} FEATURES -----------------------------------\n".format(
            number_of_features))

        text2bow = CountVectorizer(stop_words=stopwords.words('english'), max_features=number_of_features,
                                   ngram_range=(1, 3))
        text2tfidf = TfidfVectorizer(max_features=number_of_features, ngram_range=(1, 5))
#------------------------------------------------------------------Feature Representations----------------------------------------------------------------------------
        train_input_bow = bag_of_words_representation(text2bow, train_text)
        test_input_bow = bag_of_words_representation(text2bow, test_text, test=True)

        train_input_tf_idf = tf_idf_representation(text2tfidf, train_text)
        test_input_tf_idf = tf_idf_representation(text2tfidf, test_text, test=True)
        logging.info(train_input_bow.shape)

        logging.info("\n---------------Logisic Regression for Bag Of Words, 5-fold cross validation---------------------------------")

        lr = LogisticRegression()
        lr_model = lr.fit(train_input_tf_idf, np.ravel(train_labels))

        logging.info(lr_model)
        y_pred_lr = lr.predict(test_input_tf_idf)

        print("Accuracy is: ", metrics.accuracy_score(test_labels, y_pred_lr))
        print("Mean Squared Error is:", np.sqrt(mean_squared_error(test_labels, y_pred_lr)))


        logging.info("Accuracy is: ")
        logging.info(metrics.accuracy_score(test_labels, y_pred_lr))
        logging.info("Mean Squared Error is:")
        logging.info(np.sqrt(mean_squared_error(test_labels, y_pred_lr)))

        lr_cm = metrics.confusion_matrix(test_labels, y_pred_lr)
        print(lr_cm)
        logging.info(lr_cm)

        print(metrics.classification_report(test_labels, y_pred_lr))
        logging.info(metrics.classification_report(test_labels, y_pred_lr))

        # Calculating the predicted probabilities for test data
        y_pred_prob_lr = lr.predict_proba(test_input_tf_idf)[:, 1]
        logging.info(metrics.roc_auc_score(test_labels, y_pred_prob_lr))
        pickle.dump(lr, open("The_Best_Logistic_Regression_model_tfidf.pkl", 'wb'))

        labels = np.array([['3168'  ,'601'],[ '569', '2817']])
        labels = sns.heatmap(lr_cm, annot=labels, fmt='')
        mpl.show()
        labels.figure.savefig("Confusion Matrix Plot Directory\Logistic Regression(TF-idf) Confusion Matrix for 8000 features")

#----------------------------------------------------------------Plotting Confusion Matrix for best logistic regresssion model------------------------------------------------------------------------------------


#----------------------------------------------------------------------Naive Bayes-----------------------------------------------------------------------------------------------

        nb = MultinomialNB()  # one of the two classic naive Bayes variants used in text classification

        nb.fit(train_input_tf_idf, np.ravel(train_labels))
        y_pred_class = nb.predict(test_input_tf_idf)
        print("Accuracy is:", metrics.accuracy_score(test_labels, y_pred_class))
        print("Mean Squared Error is:", np.sqrt(mean_squared_error(test_labels, y_pred_class)))
        logging.info("\The Accuracy is:")
        logging.info(metrics.accuracy_score(test_labels, y_pred_class))
        logging.info("Mean Squared Error is:")
        logging.info(np.sqrt(mean_squared_error(test_labels, y_pred_class)))


        nb_cm = metrics.confusion_matrix(test_labels, y_pred_class)
        print(nb_cm)
        logging.info(nb_cm)

        print(metrics.classification_report(test_labels, y_pred_class))
        logging.info(metrics.classification_report(test_labels, y_pred_class))

        # Calculate predicted probabilities for test data
        y_pred_prob_nb = nb.predict_proba(test_input_tf_idf)[:, 1]
        logging.info(metrics.roc_auc_score(test_labels, y_pred_prob_nb))
        pickle.dump(nb, open("The_Best_Naive_Bayes_model_tfidf.pkl", 'wb'))

        #Plotting the confusion matrix....

        labels = np.array([['3213',  '556'],[ '634', '2752']])
        labels = sns.heatmap(nb_cm, annot=labels, fmt='')
        mpl.show()
        labels.figure.savefig("Confusion Matrix Plot Directory\ Naive Bayes(TF-idf) Confusion Matrix for 8000 features")

# #------------------------------------------------------------------------SVM--------------------------------------------------------------------------------------------------
        svc = SVC(kernel='linear', random_state=1)
        svc.fit(train_input_tf_idf, np.ravel(train_labels))
        y_pred_svm = svc.predict(test_input_tf_idf)

        print("Accuracy is:", metrics.accuracy_score(test_labels, y_pred_svm))
        print("Mean Squared Error is:", np.sqrt(mean_squared_error(test_labels, y_pred_svm)))
        print(metrics.classification_report(test_labels, y_pred_svm))

        logging.info("Accuracy is:")
        logging.info(metrics.accuracy_score(test_labels, y_pred_svm))
        logging.info("Mean Squared Error is:")
        logging.info(np.sqrt(mean_squared_error(test_labels, y_pred_svm)))
        logging.info(metrics.classification_report(test_labels, y_pred_svm))
        logging.info(metrics.roc_auc_score(test_labels, y_pred_svm))
        pickle.dump(svc, open("The_Best_SVM_model_tfidf.pkl", 'wb'))


        svm_cm = metrics.confusion_matrix(test_labels, y_pred_svm)
        print(svm_cm)
        logging.info(svm_cm)

#------------------------------------------------------------------------Random Forest Model-------------------------------------------------------------------------------------

        rf = RandomForestClassifier(random_state=1)
        param_grid = {
            'n_estimators': [200],
            'max_depth': [50, 60, 70]
        }
        grid_search_rf = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
        grid_search_rf.fit(train_input_tf_idf, np.ravel(train_labels))
        grid_search_rf.best_params_
        y_pred_rf = grid_search_rf.predict(test_input_tf_idf)
        print("Accuracy is:", metrics.accuracy_score(test_labels, y_pred_rf))
        logging.info("\n The Accuracy is:")
        logging.info(metrics.accuracy_score(test_labels, y_pred_rf))

        print("Mean Squared Error is:", np.sqrt(mean_squared_error(test_labels, y_pred_rf)))
        logging.info("\ The Mean Squared Error is:")
        logging.info( np.sqrt(mean_squared_error(test_labels, y_pred_rf)))


        rf_cm = metrics.confusion_matrix(test_labels, y_pred_rf)
        print(rf_cm)
        logging.info(rf_cm)

        print(metrics.classification_report(test_labels, y_pred_rf))
        logging.info(metrics.classification_report(test_labels, y_pred_rf))



        # Calculate predicted probabilities for test data
        y_pred_prob_rf = grid_search_rf.predict_proba(test_input_tf_idf)[:, 1]
        logging.info(metrics.roc_auc_score(test_labels, y_pred_prob_rf))

        pickle.dump(rf, open("The_Best_Random_Forest_model_tfidf.pkl", 'wb'))
