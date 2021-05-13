import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import autots
from autots.datasets import  autotimeseries

print("working perfectly")

df = pd.read_csv("AMZN.csv",usecols=['Date', 'Close'])
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values("Date")    

train_df = df.iloc[:2800]
test_df = df.iloc[2800:]


#visualizing the train test  values for reference
train_df.Close.plot(figsize=(15,8), title= 'AMZN Stock Price', fontsize=14, label='Train')
test_df.Close.plot(figsize=(15,8), title= 'AMZN Stock Price', fontsize=14, label='Test')
plt.legend()
plt.grid()
plt.show()


model = auto_timeseries(forecast_period=219, score_type='rmse', time_interval='D', model_type='best')
model.fit(traindata= train_df, ts_column="Date", target="Close")

model.get_leaderboard()
model.plot_cv_scores()

future_predictions = model.predict(testdata=219)
