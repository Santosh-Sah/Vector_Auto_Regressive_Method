# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 10:41:20 2020

@author: Santosh Sah
"""
from statsmodels.tools.eval_measures import rmse

from VectorAutoRegressiveMethodUtils import (importVectorAutoRegressiveMethodDataset, readVectorAutoRegressiveMethodForecastedValues)

"""

calculating VectorAutoRegressiveMethod metrics

"""
def testVectorAutoRegressiveMethodMetrics():
    
    numberOfObservation = 12
    
    #reading the full dataset
    vectorAutoRegressiveMethodDataset = importVectorAutoRegressiveMethodDataset("M2SLMoneyStock.csv", "PCEPersonalSpending.csv")

    #reading the forecasted values
    vectorAutoRegressiveMethodForecastedValues = readVectorAutoRegressiveMethodForecastedValues()    
    
    #rmse for money
    rmseForMoneyForecasting = rmse(vectorAutoRegressiveMethodDataset["Money"][-numberOfObservation:], vectorAutoRegressiveMethodForecastedValues["MoneyForecast"])
    
    #rmse for spending
    rmseForSpendingForecasting = rmse(vectorAutoRegressiveMethodDataset["Spending"][-numberOfObservation:], vectorAutoRegressiveMethodForecastedValues["SpendingForecast"])
    
    print(rmseForMoneyForecasting) #43.71049653558938
    
    print(rmseForSpendingForecasting) #37.00117516940808
    
    
if __name__ == "__main__":
    testVectorAutoRegressiveMethodMetrics()