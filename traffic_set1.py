import pandas as pd
import git

# trainning 
data_train = pd.read_csv("./trafficc_data/train.csv")

"""Index,geohash,day,timestamp,demand,RoadType,NumberofLanes,LargeVehicles,Landmarks,Temperature,Weather""" 

data_train = data_train.dropna()
#print(data_train.sample(10))

data_train["timestamp"] = pd.to_datetime(data_train["timestamp"] , format="%H:%M")
data_train["Hour"] = data_train["timestamp"].dt.hour
data_train["Minutes"] = data_train["timestamp"].dt.minute
data_train = data_train.drop(columns = ["timestamp" , "Index"])

# encoding 

from sklearn.preprocessing import LabelEncoder

encod = LabelEncoder()

cols =  [
    'geohash',
    'RoadType',
    'LargeVehicles',
    'Landmarks',
    'Weather'
]

for col in cols :
    data_train[col] = encod.fit_transform(data_train[col])

#print(data_train.sample(10))

X = data_train.drop(["demand"] , axis = 1)
Y = data_train["demand"]

df = data_train.isnull().sum()
#print(df)
#print(X.head())
#print(Y.head())

# data splitting 
from sklearn.model_selection import train_test_split
x_train ,x_test , y_train  , y_test  = train_test_split(X , Y , train_size=0.8 , random_state= 42)

"""# training
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(x_train , y_train)

y_pred = model.predict(x_test)

# val 

from sklearn.metrics import r2_score , mean_squared_error , mean_absolute_error
import numpy as np

r2 = r2_score(y_test , y_pred)
rmse = mean_squared_error(y_test , y_pred)
mae = mean_absolute_error(y_test , y_pred)

score = max(0 ,100*r2)

print(f" r2 = {r2} , rmse = {rmse} , mae = {mae} score = {score}")"""

