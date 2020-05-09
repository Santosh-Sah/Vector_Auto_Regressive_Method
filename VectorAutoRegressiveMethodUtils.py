# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:50:57 2020

@author: Santosh Sah
"""
import pandas as pd
import pickle
from statsmodels.tsa.stattools import adfuller
"""
Import dataset and read specific column. Split the dataset in training and testing set.
"""
def importVectorAutoRegressiveMethodDataset(vectorAutoRegressiveMethodDatasetFileName1, vectorAutoRegressiveMethodDatasetFileName2):
    
    vectorAutoRegressiveMethodDataset1 = pd.read_csv(vectorAutoRegressiveMethodDatasetFileName1,index_col=0, parse_dates=True)
    
    #the dataset is minthly dataset. Hence setting its frequency as monthly.
    vectorAutoRegressiveMethodDataset1.index.freq = "MS"
    
    vectorAutoRegressiveMethodDataset2 = pd.read_csv(vectorAutoRegressiveMethodDatasetFileName2,index_col=0, parse_dates=True)
    
    #the dataset is minthly dataset. Hence setting its frequency as monthly.
    vectorAutoRegressiveMethodDataset2.index.freq = "MS"
    
    vectorAutoRegressiveMethodDatasetFinalDataset = vectorAutoRegressiveMethodDataset1.join(vectorAutoRegressiveMethodDataset2)
    
    return vectorAutoRegressiveMethodDatasetFinalDataset

#splitting dataset into training and testing set
def splitVectorAutoRegressiveMethodDataset(vectorAutoRegressiveMethodDataset):
    
    #splitting the dataset into training and testing set.
    vectorAutoRegressiveMethodTrainingSet = vectorAutoRegressiveMethodDataset.iloc[0:-12]
    vectorAutoRegressiveMethodTestingSet = vectorAutoRegressiveMethodDataset.iloc[-12:]
    
    return vectorAutoRegressiveMethodTrainingSet, vectorAutoRegressiveMethodTestingSet

"""
Save training and testing dataset
"""
def saveTrainingAndTestingDataset(X_train, X_test):
    
    #Write X_train in a picke file
    with open("X_train.pkl",'wb') as X_train_Pickle:
        pickle.dump(X_train, X_train_Pickle, protocol = 2)
    
    #Write X_test in a picke file
    with open("X_test.pkl",'wb') as X_test_Pickle:
        pickle.dump(X_test, X_test_Pickle, protocol = 2)

"""
read X_train from pickle file
"""
def readVectorAutoRegressiveMethodXTrain():
    
    #load X_train
    with open("X_train.pkl","rb") as X_train_pickle:
        X_train = pickle.load(X_train_pickle)
    
    return X_train

"""
read X_test from pickle file
"""
def readVectorAutoRegressiveMethodXTest():
    
    #load X_test
    with open("X_test.pkl","rb") as X_test_pickle:
        X_test = pickle.load(X_test_pickle)
    
    return X_test

"""
Save VectorAutoRegressiveMethod as a pickle file.
"""
def saveVectorAutoRegressiveMethodModel(vectorAutoRegressiveMethodModel):
    
    #Write VectorAutoRegressiveMethodModel as a picke file
    with open("VectorAutoRegressiveMethodModel.pkl",'wb') as vectorAutoRegressiveMethodModel_Pickle:
        pickle.dump(vectorAutoRegressiveMethodModel, vectorAutoRegressiveMethodModel_Pickle, protocol = 2)

"""
read VectorAutoRegressiveMethod from pickle file
"""
def readVectorAutoRegressiveMethodModel():
    
    #load VectorAutoRegressiveMethodModel model
    with open("VectorAutoRegressiveMethodModel.pkl","rb") as vectorAutoRegressiveMethodModel:
        vectorAutoRegressiveMethodModel = pickle.load(vectorAutoRegressiveMethodModel)
    
    return vectorAutoRegressiveMethodModel

"""
Save VectorAutoRegressiveMethod as a pickle file.
"""
def saveVectorAutoRegressiveMethodModelForFullDataset(vectorAutoRegressiveMethodModelForFullDataset):
    
    #Write VectorAutoRegressiveMethodModelForFullDataset as a picke file
    with open("VectorAutoRegressiveMethodModelForFullDataset.pkl",'wb') as vectorAutoRegressiveMethodModelForFullDataset_Pickle:
        pickle.dump(vectorAutoRegressiveMethodModelForFullDataset, vectorAutoRegressiveMethodModelForFullDataset_Pickle, protocol = 2)

"""
read VectorAutoRegressiveMethod from pickle file
"""
def readVectorAutoRegressiveMethodModelForFullDataset():
    
    #load VectorAutoRegressiveMethodModelForFullDataset model
    with open("VectorAutoRegressiveMethodModelForFullDataset.pkl","rb") as vectorAutoRegressiveMethodModelForFullDataset:
        vectorAutoRegressiveMethodModelForFullDataset = pickle.load(vectorAutoRegressiveMethodModelForFullDataset)
    
    return vectorAutoRegressiveMethodModelForFullDataset

"""
save VectorAutoRegressiveMethodPredictedValues as a pickle file
"""

def saveVectorAutoRegressiveMethodPredictedValues(vectorAutoRegressiveMethodPredictedValues):
    
    #Write VectorAutoRegressiveMethodPredictedValues in a picke file
    with open("VectorAutoRegressiveMethodPredictedValues.pkl",'wb') as vectorAutoRegressiveMethodPredictedValues_Pickle:
        pickle.dump(vectorAutoRegressiveMethodPredictedValues, vectorAutoRegressiveMethodPredictedValues_Pickle, protocol = 2)

"""
read VectorAutoRegressiveMethodPredictedValues from pickle file
"""
def readVectorAutoRegressiveMethodPredictedValues():
    
    #load VectorAutoRegressiveMethodPredictedValues
    with open("VectorAutoRegressiveMethodPredictedValues.pkl","rb") as vectorAutoRegressiveMethodPredictedValues_pickle:
        vectorAutoRegressiveMethodPredictedValues = pickle.load(vectorAutoRegressiveMethodPredictedValues_pickle)
    
    return vectorAutoRegressiveMethodPredictedValues

"""
save VectorAutoRegressiveMethodForecastedValues as a pickle file
"""

def saveVectorAutoRegressiveMethodForecastedValues(vectorAutoRegressiveMethodForecastedValues):
    
    #Write VectorAutoRegressiveMethodForecastedValues in a picke file
    with open("VectorAutoRegressiveMethodForecastedValues.pkl",'wb') as vectorAutoRegressiveMethodForecastedValues_Pickle:
        pickle.dump(vectorAutoRegressiveMethodForecastedValues, vectorAutoRegressiveMethodForecastedValues_Pickle, protocol = 2)

"""
read VectorAutoRegressiveMethodForecastedValues from pickle file
"""
def readVectorAutoRegressiveMethodForecastedValues():
    
    #load VectorAutoRegressiveMethodForecastedValues
    with open("VectorAutoRegressiveMethodForecastedValues.pkl","rb") as vectorAutoRegressiveMethodForecastedValues_pickle:
        vectorAutoRegressiveMethodForecastedValues = pickle.load(vectorAutoRegressiveMethodForecastedValues_pickle)
    
    return vectorAutoRegressiveMethodForecastedValues

#test dataset is stationary or non stationary
def agumentedDickeyFullerTest(series,title=''):
    
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    print(f'Augmented Dickey-Fuller Test: {title}')
    result = adfuller(series.dropna(),autolag='AIC') # .dropna() handles differenced data
    
    labels = ['ADF test statistic','p-value','# lags used','# observations']
    out = pd.Series(result[0:4],index=labels)

    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
    print(out.to_string())          # .to_string() removes the line "dtype: float64"
    
    if result[1] <= 0.05:
        print("Strong evidence against the null hypothesis")
        print("Reject the null hypothesis")
        print("Data has no unit root and is stationary")
    else:
        print("Weak evidence against the null hypothesis")
        print("Fail to reject the null hypothesis")
        print("Data has a unit root and is non-stationary")


