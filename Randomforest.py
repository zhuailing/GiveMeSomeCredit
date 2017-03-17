#!/usr/bin/env python
# coding: utf-8 
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import Imputer
#*****************************************************************************************
#Use RF model Only:
#1.load data (training and test) and preprocessing data(replace NA,98,96,0(age) with NaN)
#2.split training data into training_new  and test_new (for validation model)

#3.impute the data with imputer: replace MVs with Mean
#4.Build RF model using the training_new data:
#   a. handle imbalanced data distribution
#   b. perform parameter tuning using grid search with CrossValidation
#   c. output the best model and make predictions for test data
#*****************************************************************************************

#*****************************************************************************************
# a few tools
#*****************************************************************************************

#*****************************************************************************************
# func to create a new dict using  keys and values
# input: keys =[]and values=[]
# output: dict{}
#*****************************************************************************************
def creatDictKV(keys, vals):
    lookup = {}
    if len(keys) == len(vals):
        for i in range(len(keys)):
            key = keys[i]
            val = vals[i]
            lookup[key] = val
    #print lookup
    return lookup

#*****************************************************************************************
#compute AUC
# input: y_true =[] and y_score=[]
# output: auc
#*************************************************************************
def computeAUC(y_true, y_score):

    auc = roc_auc_score(y_true, y_score)
    print "auc= ", auc
    return auc
#*****************************************************************************************

#*****************************************************************************************
# Real Stuff:
#*****************************************************************************************

def main():
    #*************************************************************************************
    #1.load data (training and test) and preprocessing data(replace NA,98,96,0(age) with NaN)
    #read data using pandas
    #replace 98, 96 with NAN for NOTime30-59,90,60-90
    #replace  0 with NAN for age
    #*************************************************************************************
    colnames = ['ID', 'label', 'RUUnsecuredL', 'age', 'NOTime30-59', \
                'DebtRatio', 'Income', 'NOCredit', 'NOTimes90', \
                'NORealEstate', 'NOTime60-89', 'NODependents']
    col_nas = ['', 'NA', 'NA', 0, [98, 96], 'NA', 'NA', 'NA', \
                [98, 96], 'NA', [98, 96], 'NA']
    col_na_values = creatDictKV(colnames, col_nas)

    dftrain = pd.read_csv("cs-training.csv", names=colnames, \
                          na_values=col_na_values, skiprows=[0])
    train_id = [int(x) for x in dftrain.pop("ID")]
    y_train = np.asarray([int(x)for x in dftrain.pop("label")])
    x_train = dftrain.as_matrix()

    dftest = pd.read_csv("cs-test.csv", names=colnames, \
                         na_values=col_na_values, skiprows=[0])
    test_id = [int(x) for x in dftest.pop("ID")]
    y_test = np.asarray(dftest.pop("label"))
    x_test = dftest.as_matrix()

    #*************************************************************************************
    #2.split training data into training_new  and test_new (for validation model)
    # to keep the class ratio using StratifiedShuffleSplit to do the split
    #*************************************************************************************

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.33333, random_state=0)
    for train_index, test_index in sss.split(x_train, y_train):
        print("TRAIN:", train_index, "TEST:", test_index)
        x_train_new, x_test_new = x_train[train_index], x_train[test_index]
        y_train_new, y_test_new = y_train[train_index], y_train[test_index]

    y_train = y_train_new
    x_train = x_train_new

    #*****************************************************************************************
    #3.impute the data with imputer: replace MVs with Mean
    #*****************************************************************************************
    imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
    imp.fit(x_train)
    x_train = imp.transform(x_train)
    x_test_new = imp.transform(x_test_new)
    x_test = imp.transform(x_test)

    #*****************************************************************************************
    #4.Build RF model using the training_new data:
    #   a. handle imbalanced data distribution by
    #      setting class_weight="balanced"/"balanced_subsample"
    #      n_samples / (n_classes * np.bincount(y))
    #*****************************************************************************************
    #  Initialize the model:
    #*****************************************************************************************
    rf = RandomForestClassifier(n_estimators=100, \
                                oob_score=True, \
                                min_samples_split=2, \
                                min_samples_leaf=50, \
                                n_jobs=-1, \
                                #class_weight="balanced",\
                                class_weight="balanced_subsample", \
                                bootstrap=True\
                                ) 
    #*************************************************************************************
    #   b. perform parameter tuning using grid search with CrossValidation
    #*************************************************************************************

    #param_grid={"max_features": [2,3,4,5],\
	#	 "min_samples_leaf": [30,40,50,100],\
	#	 "criterion": ["gini", "entropy"]}
    param_grid = {"max_features": [2, 3, 4], "min_samples_leaf":[50]}
    grid_search = GridSearchCV(rf, cv=10, scoring='roc_auc', param_grid=param_grid, iid=False)

    #*************************************************************************************
    #   c. output the best model and make predictions for test data
    #       - Use best parameter to build model with training_new data
    #*************************************************************************************
    grid_search.fit(x_train, y_train)
    print "the best parameter:", grid_search.best_params_
    print "the best score:", grid_search.best_score_
    #print "the parameters used:",grid_search.get_params

    #*************************************************************************************
    #   To see how fit the model with the training_new data
    #       -Use the model trained to make predication for train_new data
    #*************************************************************************************

    predicted_probs_train = grid_search.predict_proba(x_train)
    predicted_probs_train = [x[1] for  x in predicted_probs_train]
    computeAUC(y_train, predicted_probs_train)

    #*************************************************************************************
    #   To see how well the model performs with the test_new data
    #    -Use the model trained to make predication for validataion data (test_new)
    #*************************************************************************************
    predicted_probs_test_new = grid_search.predict_proba(x_test_new)
    predicted_probs_test_new = [x[1] for x in predicted_probs_test_new]
    computeAUC(y_test_new, predicted_probs_test_new)

    #*************************************************************************************
    #  use the model to predict for test and output submission file
    #*************************************************************************************
    predicted_probs_test = grid_search.predict_proba(x_test)
    predicted_probs_test = ["%.9f" % x[1] for x in predicted_probs_test]
    submission = pd.DataFrame({'ID':test_id, 'Probabilities':predicted_probs_test})
    submission.to_csv("rf_benchmark.csv", index=False)

#*************************************************************************************
if __name__ == "__main__":
    main()

