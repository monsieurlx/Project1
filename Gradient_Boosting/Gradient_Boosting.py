#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[2]:

import pandas as pd 
import numpy as np
import sys
#feature scaling 
from sklearn.preprocessing import StandardScaler
#metrics
from sklearn import metrics

#mlflow
import mlflow
import mlflow.sklearn
# let's create test and validation set 
from sklearn.model_selection import train_test_split
#Gradient boosting 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
#Metrics function
def eval_metrics(actual, pred):
    rmse = np.sqrt(metrics.mean_squared_error(actual, pred))
    mae = metrics.mean_absolute_error(actual, pred)
    r2 = metrics.r2_score(actual, pred)
    return rmse, mae, r2
    

df = pd.read_csv('/home/leo/Python/proj1/prep_application_train.csv')

#Clean the data by replacing NaN with zero
df2 = np.nan_to_num(df)

#Prepare data

X = df2[:, 1:13]
y = df2[:, 0]

sc = StandardScaler()
X = sc.fit_transform(X)

X_trains, X_tests, y_trains, y_tests = train_test_split(X, y, test_size=0.2, random_state=0)

estim = int(sys.argv[1]) if len(sys.argv) > 1 else 1
lr = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5    
mf = int(sys.argv[3]) if len(sys.argv) > 3 else 10
dep = int(sys.argv[4]) if len(sys.argv) > 4 else 10 
rand = int(sys.argv[5]) if len(sys.argv) > 5 else 5

with mlflow.start_run():

    
    gb_clf2 = GradientBoostingClassifier(n_estimators=estim, learning_rate= lr, max_features=mf, max_depth=dep, random_state=rand)
    gb_clf2.fit(X_trains, y_trains)
    predictions = gb_clf2.predict(X_tests)
    y_pred = gb_clf2.predict(X_tests)
           
    
    print("Gradient Boosting (n_estimators=%f, learning_rate=%f,max_features=%f, max_depth=%f, random_states=%f)):" % (estim, lr,mf,dep,rand))


    
    (rmse, mae, r2) = eval_metrics(y_tests, y_pred)
    # Print out metrics
    print('Mean Absolute Error:', rmse)
    print('Mean Squared Error:', mae)
    print('Root Mean Squared Error:', r2)
    # Log parameter, metrics, and model to MLflow
 # mlflow.log_param("n_estimators", estim)
 # mlflow.log_param("learning_rate", lr)
 # mlflow.log_param("max_features", mf)
 # mlflow.log_param("max_depth", dep)
 # mlflow.log_param("random state", rand)
    mlflow.log_metric("mae", mae)
    mlflow.log_metric("r2", r2)
    mlflow.log_metric("rmse", rmse)
    
    
    mlflow.sklearn.log_model(gb_clf2, "model")

