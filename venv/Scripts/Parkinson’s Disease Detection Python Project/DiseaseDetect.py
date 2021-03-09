import numpy as np
import pandas as pd
import os
import sys
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#load the dataste in pandas dataframe
def load_data (input_file):
    df = pd.read_csv(input_file)
    print("Disease Detection")
    print(df.shape)
    print(df.head())
    print("end")

#get the features and labels from the Dataframe
    features = df.loc[:, df.columns != 'status'].values[:, 1:]
    labels = df.loc[:, 'status'].values

#The ‘status’ column has values 0 and 1 as labels; let’s get the counts of these labels for both- 0 and 1.
    print(labels[labels == 1].shape[0], labels[labels == 0].shape[0])

#Initialize a MinMaxScaler and scale the features to between -1 and 1 to normalize them.
    # The MinMaxScaler transforms features by scaling them to a given range.
 # The fit_transform() method fits to the data and then transforms it. We don’t need to scale the labels

    scaler = MinMaxScaler((-1, 1))
    x = scaler.fit_transform(features)
    y = labels

#split the dataset into training and testing sets keeping 20% of the data for testing

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=7)

#Initialize the XGBClassifier and training the model
    model = XGBClassifier()
    model.fit(x_train, y_train)

#Finally, generate y_pred (predicted values for x_test) and calculate the accuracy for the model. Print it out.
    y_pred = model.predict(x_test)
    print(accuracy_score(y_test, y_pred) * 100)
    return  df

#Get the features an labels from the DataFrame (dataset).
# The features are all the columns except ‘status’, and the labels are those in the ‘status’ column.

if __name__ == "__main__":
    input_file = 'parkinsons.data'
    df = load_data(input_file)
