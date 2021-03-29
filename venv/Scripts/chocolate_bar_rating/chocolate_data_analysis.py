import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import sys


def load_data(input_csv_file):
    df = pd.read_csv(input_csv_file)
    print(df.head())

    #changing the column name
    df.columns = ['Company_Name','Country_of_origin', 'REF', 'Review_Date', 'Cocoa_percent', 'Company_Location', 'Rating', 'Bean_Type','Broad_Bean_Origin']
    print(df.shape)

    #checking the dataset information
    print(df.info())

    #handling the missing values
    df = df.replace('?', np.NaN)
    missing  = df.isna().sum()

    df.fillna('UNKNOWN', inplace=True)
    print(df.isnull().sum())

    print(df.shape)
    print(df)

    print(df.dtypes)

    df["Company"] = df["Company_Name"].astype('category')
    df['Cocoa_percent'] = (df['Cocoa_percent']).str.replace('%', ' ')
    df['Cocoa_percent'] = (df['Cocoa_percent']).astype(float)

    print(df.dtypes)

    print(df.describe(include ='all'))

if __name__ == "__main__":
    input_csv_file = "flavors_of_cacao.csv"
    df = load_data(input_csv_file)