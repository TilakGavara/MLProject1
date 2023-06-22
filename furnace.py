import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from datetime import datetime, timedelta

data= "dataset.csv"
df = pd.read_csv(data)
df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])
df['DATE_TIME'] = df['DATE_TIME'].apply(lambda x: x.timestamp())
X = df[['DATE_TIME', 'CB_FLOW', 'CB_PRESS', 'CB_TEMP', 'STEAM_FLOW', 'STEAM_TEMP', 'STEAM_PRESS','O2_PRESS', 'O2_FLOW', 'O2_PER', 'PCI', 'ATM_HUMID', 'HB_TEMP', 'HB_PRESS', 'TOP_PRESS','TOP_TEMP1', 'TOP_SPRAY', 'TOP_TEMP', 'TOP_PRESS_1', 'H2', 'CO', 'CO2']]
y=df[['CO/CO2 RATIO','CO/CO2_RATIO_AFTER_1_HOUR','CO/CO2_RATIO_AFTER_2_HOURS','CO/CO2_RATIO_AFTER_3_HOURS','CO/CO2_RATIO_AFTER_4_HOURS']]


column_mean = df['CO/CO2_RATIO_AFTER_1_HOUR'].mean()
df['CO/CO2_RATIO_AFTER_1_HOUR'].fillna(column_mean, inplace=True)
column_mean1 = df['CO/CO2_RATIO_AFTER_2_HOURS'].mean()
df['CO/CO2_RATIO_AFTER_2_HOURS'].fillna(column_mean1, inplace=True)
column_mean2 = df['CO/CO2_RATIO_AFTER_3_HOURS'].mean()
df['CO/CO2_RATIO_AFTER_3_HOURS'].fillna(column_mean2, inplace=True)
column_mean3 = df['CO/CO2_RATIO_AFTER_4_HOURS'].mean()
df['CO/CO2_RATIO_AFTER_4_HOURS'].fillna(column_mean3, inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models=[]
for i in range(5):
  model = RandomForestRegressor(n_estimators=100, random_state=42)
  model.fit(X_train, y_train.iloc[:,i])
  models.append(model)

predictions = []
for i in range(5):
    y_pred = models[i].predict(X_test)
    predictions.append(y_pred)
for i in range(5):
  mse = mean_squared_error(y_test.iloc[:,i], predictions[i])
  print(y_test.iloc[:,i])
  print(predictions[i])
  print(f"Mean Squared Error: {mse}")