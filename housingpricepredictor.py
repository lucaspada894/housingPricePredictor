#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  8 19:50:00 2020

@author: lucaspada894
"""
 
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from matplotlib import pyplot as plt
from itertools import combinations 
import statsmodels.api as sm
from sklearn.linear_model import Ridge

reg = LinearRegression()

housing_df = pd.read_csv("housingdata.csv")
housing_df.head()
#df_x = pd.DataFrame(housing_df, columns = housing_df.feature_names)
CRIM = housing_df['CRIM']
ZN = housing_df['ZN']
INDUS = housing_df['INDUS']
CHAS = housing_df['CHAS']
NOX = housing_df['NOX']
RM = housing_df['RM']
AGE = housing_df['AGE']
DIS = housing_df['DIS']
RAD = housing_df['RAD']
TAX = housing_df['TAX']
PTRATIO = housing_df['PTRATIO']
B = housing_df['B']
LSTAT = housing_df['LSTAT']
MEDV = housing_df['MEDV']

plt.figure(1)
plt.scatter(CRIM,MEDV)
plt.xlabel("CRIM")
plt.ylabel("MEDV")
plt.title("CRIM")

plt.figure(2)
plt.scatter(ZN,MEDV)
plt.xlabel("ZN")
plt.ylabel("MEDV")
plt.title("ZN")

plt.figure(3)
plt.scatter(INDUS,MEDV)
plt.xlabel("INDUS")
plt.ylabel("MEDV")
plt.title("INDUS")

plt.figure(4)
plt.scatter(CHAS,MEDV)
plt.xlabel("CHAS")
plt.ylabel("MEDV")
plt.title("CHAS")

plt.figure(5)
plt.scatter(NOX,MEDV)
plt.xlabel("NOX")
plt.ylabel("MEDV")
plt.title("NOX")

plt.figure(6)
plt.scatter(RM,MEDV)
plt.xlabel("RM")
plt.ylabel("MEDV")
plt.title("RM")

plt.figure(7)
plt.scatter(AGE,MEDV)
plt.xlabel("AGE")
plt.ylabel("MEDV")
plt.title("AGE")

plt.figure(8)
plt.scatter(DIS,MEDV)
plt.xlabel("DIS")
plt.ylabel("MEDV")
plt.title("DIS")

plt.figure(9)
plt.scatter(RAD,MEDV)
plt.xlabel("RAD")
plt.ylabel("MEDV")
plt.title("RAD")

plt.figure(10)
plt.scatter(TAX,MEDV)
plt.xlabel("TAX")
plt.ylabel("MEDV")
plt.title("TAX")

plt.figure(11)
plt.scatter(PTRATIO,MEDV)
plt.xlabel("PTRATIO")
plt.ylabel("MEDV")
plt.title("PTRATIO")

plt.figure(12)
plt.scatter(B,MEDV)
plt.xlabel("B")
plt.ylabel("MEDV")
plt.title("B")

plt.figure(13)
plt.scatter(LSTAT,MEDV)
plt.xlabel("LSTAT")
plt.ylabel("MEDV")
plt.title("LSTAT")

X = housing_df[["AGE","INDUS","NOX","RM","TAX"]]
y = housing_df[["MEDV"]]

" GETTING BEST MODELS"
X_train = X.head(400)
X_test = X.tail(106)
y_train = y.head(400)
y_test = y.tail(106)

best_model_features = []

def processSubset(feature_set, x_train_samples, y_train_samples):
    # Fit model on feature_set and calculate RSS
    regr = LinearRegression();
    model = regr.fit(x_train_samples[list(feature_set)],y_train_samples)
    y_predict = model.predict(x_train_samples[list(feature_set)])
    MSE = mean_squared_error(y_train_samples,y_predict)
    return {"model":model, "MSE":MSE, "features":feature_set}

def getBest(k, x_train_samples, y_train_samples, best_features_arr):
        
    results = []
    
    for combo in combinations(x_train_samples.columns, k):
        results.append(processSubset(combo, x_train_samples, y_train_samples))
    
    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)
    
    # Choose the model with the lowest MSE
    best_model = models.loc[models['MSE'].idxmin()]
    best_features_arr.append(best_model['features'])
        
    # Return the best model, along with some other useful information about the model
    return best_model

best_models = []
predicted_validation_MSE = []

for i in range(1,6):
    best_models.append(getBest(i,X_train,y_train, best_model_features))
    
"GET PREDICTED Y WITH MODELS"
for i in range(0,len(best_models)):
    y_pred = best_models[i]["model"].predict(X_test[list(best_model_features[i])])
    model_mse = mean_squared_error(y_test,y_pred)
    predicted_validation_MSE.append(model_mse) 
    
print("best model features:",best_model_features)

MSE_for_plot = []

for i in range(0,5):
    MSE_for_plot.append(best_models[i].loc['MSE'])
    
x_axis = [1,2,3,4,5,]
    
    
plt.figure(14)
plt.title("Training loss - best subsets")
plt.xlabel("number of features")
plt.ylabel("best model MSE")
plt.plot(x_axis,MSE_for_plot,color="red")

plt.figure(15)
plt.title("Validation loss - best subsets")
plt.xlabel("number of features")
plt.ylabel("best model MSE")
plt.plot(x_axis,predicted_validation_MSE,color="red")
    
"Part iii"
best_features_all = []
best_models_all = []
best_models_all_MSE = []
"Fitting to all samples"
for i in range(1,6):
    best_models_all.append(getBest(i,X,y,best_features_all))
    
"get new MSEs"
for i in range(0,len(best_models_all)):
    y_pred = best_models_all[i]["model"].predict(X[list(best_features_all[i])])
    model_mse = mean_squared_error(y,y_pred)
    best_models_all_MSE.append(model_mse) 

"get Cp" 
Cp = [] 
for i in range(0,len(best_models_all)):
    mse = best_models_all_MSE[i]
    mseAllFeatures = best_models_all_MSE[len(best_models_all_MSE)-1]
    Cp.append(mse + ((2*mseAllFeatures)/506)*(i+1))

plt.figure(16)
plt.title("Mallow's Cp - best subsets")
plt.xlabel("number of features")
plt.ylabel("best model MSE")
plt.plot(x_axis,Cp,color="red")    

"Ridge regression part c iv"

original_mse = predicted_validation_MSE[4]

new_features = ["AGE","INDUS","NOX","RM","TAX"]

"get l2 norm"


def getL2Coef(reg):
    totSum = 0
    for i in range(0,len(reg.coef_[0])):
        totSum += reg.coef_[0][i]**2
        
    return totSum

def getL1Coef(reg):
    totSum = 0
    for i in range(0,len(reg.coef_[0])):
        totSum += abs(reg.coef_[0][i])
        
    return totSum

    

y_ridge_vals = []
iterations = 0
min_ridge = original_mse
alpha = 1.0


while(1):
    clf = Ridge(alpha=alpha)
    modelRidge = clf.fit(X_train[list(new_features)], y_train)
    y_pred_ridge = modelRidge.predict(X_test)
    mse = mean_squared_error(y_test,y_pred_ridge)
    if mse < original_mse and iterations < 20:
        y_ridge_vals.append(mse)
        alpha +=  1.0
        iterations += 1
    else:
        break
    "complexity = mse + alpha*getL2Coef(model)"
       
        
x_ridge_axis = np.linspace(0,alpha,iterations)


plt.figure(17)
plt.title("ridge validation")
plt.xlabel("lambda")
plt.ylabel("ridge MSE")
plt.plot(x_ridge_axis,y_ridge_vals,color="red")  

"LASSO"

x_lasso_axis = []
y_lasso_vals = []

iterations = 0
lasso_min = original_mse
alpha = 0.05
while(1):
    clf = linear_model.Lasso(alpha=alpha)
    modelLasso = clf.fit(X_train[list(new_features)], y_train)
    y_pred_ridge = modelLasso.predict(X_test)
    mse = mean_squared_error(y_test,y_pred_ridge)
    if mse < original_mse:
        lasso_min = mse
        y_lasso_vals.append(lasso_min)
        alpha +=  0.05
        iterations += 1
    else:
        break
    
x_lasso_axis = np.linspace(0,alpha,iterations)

plt.figure(18)
plt.title("lasso validation")
plt.xlabel("lambda")
plt.ylabel("Lasso MSE")
plt.plot(x_lasso_axis,y_lasso_vals,color="red")     


