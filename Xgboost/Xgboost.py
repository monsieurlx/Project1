#!/usr/bin/env python
# coding: utf-8


#!/usr/bin/env python
# coding: utf-8

import pandas as pd 
import numpy as np
import sys
#feature scaling 
from sklearn.preprocessing import StandardScaler
#metrics
from sklearn import metrics
from xgboost import XGBClassifier
#mlflow
import mlflow
import mlflow.sklearn
# let's create test and validation set 
from sklearn.model_selection import train_test_split
#Gradient boosting 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
#Metrics function
def eval_metrics(actual, pred):
    rmse = np.sqrt(metrics.mean_squared_error(actual, pred))
    mae = metrics.mean_absolute_error(actual, pred)
    r2 = metrics.r2_score(actual, pred)
    return rmse, mae, r2
    

df = pd.read_csv('/home/leo/Python/MLFLOW/prep_application_train.csv')

#Clean the data by replacing NaN with zero
df2 = np.nan_to_num(df)

#Prepare data

X = df2[:, 1:13]
y = df2[:, 0]

sc = StandardScaler()
X = sc.fit_transform(X)

X_trains, X_tests, y_trains, y_tests = train_test_split(X, y, test_size=0.2, random_state=0)

gam = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
lr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5   
dep = int(sys.argv[3]) if len(sys.argv) > 3 else 5 
mds = int(sys.argv[4]) if len(sys.argv) > 4 else 5
lb = float(sys.argv[5]) if len(sys.argv) > 5 else 1
al = float(sys.argv[6]) if len(sys.argv) > 6 else 0

with mlflow.start_run():

    
    xgb_clf = XGBClassifier(min_split_loss = gam ,learning_rate = lr,max_depth = dep, max_delta_step=mds, reg_lambda=lb, reg_alpha = al)
    xgb_clf.fit(X_trains, y_trains)
    y_pred = xgb_clf.predict(X_tests)
           
    
    print("Gradient Boosting (gamma=%f, learning_rate=%f,max_depth=%f,max_delta_step=%f lambda=%f, alpha=%f)):" % (gam, lr,dep,mds,lb,al))


    
    (rmse, mae, r2) = eval_metrics(y_tests, y_pred)
    # Print out metrics
    print('Mean Absolute Error:', rmse)
    print('Mean Squared Error:', mae)
    print('Root Mean Squared Error:', r2)
    # Log parameter, metrics, and model to MLflow
    mlflow.log_param("n_estimators", gam)
    mlflow.log_param("random state", lr)
    mlflow.log_metric("rmse", dep)
    mlflow.log_metric("r2", mds)
    mlflow.log_metric("mae", lb)
    mlflow.log_metric("r2", al)
    
    mlflow.sklearn.log_model(xgb_clf, "model")

