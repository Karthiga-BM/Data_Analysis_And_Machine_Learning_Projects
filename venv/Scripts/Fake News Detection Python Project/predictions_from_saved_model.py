import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
import logging
import pandas as pd

# Model predictions with pickle models

logging.basicConfig(handlers=[logging.FileHandler('logistic_regression_model_prediction_analysis_log.txt', 'w', 'utf-8')], level=logging.INFO,
                    format='%(asctime)s %(message)s')



def load_data(json_file):


    dataset = pd.read_json(json_file)
    text = list(dataset['headline'])
    labels = list(dataset['is_sarcastic'])

    if len(text) != len(labels):
        exit("[ERROR]: there is a different number of input text strings and output labels")
    return text, labels

def tf_idf_representation(text2tfidf, headlines_list, test=False):

    if test:
        input_fts = text2tfidf.transform(headlines_list).toarray()
    else:
        input_fts = text2tfidf.fit_transform(headlines_list).toarray()

    return input_fts

train_text, train_labels = load_data('train.json')
test_text, test_labels = load_data('test.json')

text2tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 3))

train_input_fts_tf_idf = tf_idf_representation(text2tfidf, train_text)
test_input_fts_tf_idf = tf_idf_representation(text2tfidf, test_text, test=True)

model_to_load="best_logistic_regression_model.pkl"
loaded_model = pickle.load(open(model_to_load, 'rb')) #load pickle model
result = loaded_model.score(test_input_fts_tf_idf, test_labels) #get model accuracy on the test set
logging.info("Loaded model {}\n model score={}".format(model_to_load, result))
predictions = loaded_model.predict(test_input_fts_tf_idf) #make predictions

predictions_li = predictions.tolist()
cm = confusion_matrix(test_labels, predictions_li) #generate confusion matrix
logging.info("Confusion matrix")
logging.info(cm)


predictions_different_from_labels = {}
for i in range(len(test_text)):
    if test_labels[i] != predictions_li[i]:
        predictions_different_from_labels[test_text[i]] = [str(test_labels[i]), str(predictions_li[i])]

counter = 0
logging.info("Misclassified sentence-label pairs (format: sentence, true label, predicted label)")
for k,v in predictions_different_from_labels.items(): #log headlines misclassified headlines, their true labels and their predicted labels
    counter += 1
    logging.info("{} {}".format(k, ' '.join(v)))
logging.info("Total number of misclassified headlines: {}".format(counter))
