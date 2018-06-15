# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 16:54:02 2018

@author: Tathagat Dasgupta
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

df=pd.read_csv("cal_housing_clean.csv")
print(df.describe()) #to understand the dataset

y_val= df["medianHouseValue"]
x_data=df.drop("medianHouseValue",axis=1)


X_train, X_eval,y_train,y_eval=train_test_split(x_data,y_val,test_size=0.3,random_state=101)


scaler_model = MinMaxScaler()
scaler_model.fit(X_train)

X_train=pd.DataFrame(scaler_model.transform(X_train),columns=X_train.columns,index=X_train.index)

scaler_model.fit(X_eval)

X_eval=pd.DataFrame(scaler_model.transform(X_eval),columns=X_eval.columns,index=X_eval.index)

#Creating Feature Columns
feat_cols=[]
for cols in df.columns[:-1]:
    column=tf.feature_column.numeric_column(cols)
    feat_cols.append(column)
    
print(feat_cols)

#The estimator model
model=tf.estimator.DNNRegressor(hidden_units=[6,10,6],feature_columns=feat_cols)

#the input function
input_func=tf.estimator.inputs.pandas_input_fn(X_train,y_train,batch_size=10,num_epochs=1000,shuffle=True)

#Training the model
model.train(input_fn=input_func,steps=1000)

#Evaluating the model
train_metrics=model.evaluate(input_fn=input_func,steps=1000)

#Now to predict values we do the following
pred_input_func=tf.estimator.inputs.pandas_input_fn(x=X_eval,y=y_eval,batch_size=10,num_epochs=1,shuffle=False)
preds=model.predict(input_fn=pred_input_func)

predictions=list(preds)
final_pred=[]
for pred in predictions:
    final_pred.append(pred["predictions"])
    
test_metric=model.evaluate(input_fn=pred_input_func,steps=1000)    

