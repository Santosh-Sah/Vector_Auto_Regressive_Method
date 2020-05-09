# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:52:22 2020

@author: Santosh Sah
"""
import pandas as pd
from VectorAutoRegressiveMethodUtils import (readVectorAutoRegressiveMethodModel, 
                                               readVectorAutoRegressiveMethodXTrain,
                                               saveVectorAutoRegressiveMethodForecastedValues,
                                               importVectorAutoRegressiveMethodDataset, 
                                               readVectorAutoRegressiveMethodForecastedValues)

from VectorAutoRegressiveMethodVisualization import (visualizeVectorAutoRegressiveMethodPredictedValuesForMoney,
                                                     visualizeVectorAutoRegressiveMethodPredictedValuesForSpending, 
                                                     visualizeResultPlots, visualizeForecastedPlots)

"""
test the model on testing dataset
"""
def testVectorAutoRegressiveMethodModel():
    
    #reading the full dataset
    vectorAutoRegressiveMethodDataset = importVectorAutoRegressiveMethodDataset("M2SLMoneyStock.csv", "PCEPersonalSpending.csv")
    
    #reading testing data
    X_train = readVectorAutoRegressiveMethodXTrain()
    
    #reading model from pickle file
    vectorAutoRegressiveMethodModel = readVectorAutoRegressiveMethodModel()
    
    #Unlike the VARMAX model we'll use in upcoming sections, the VAR .forecast() function 
    #requires that we pass in a lag order number of previous observations as well. 
    #Unfortunately this forecast tool doesn't provide a DateTime index - we'll have to do that manually.
    #forecast for next 12 months
    predictedValues = vectorAutoRegressiveMethodModel.forecast(y = X_train.values[-5:], steps = 12)
    
    
    idx = pd.date_range('1/1/2015', periods=12, freq='MS')
    
    vectorAutoRegressiveMethodForecastedValues = pd.DataFrame(predictedValues, index=idx, columns=['Money2d','Spending2d'])
    
    numberOfObsevation = 12
    
    #Invert the transformation
    #Remember that the forecasted values represent second-order differences. 
    #To compare them to the original data we have to roll back each difference. 
    #To roll back a first-order difference we take the most recent value on the training side of the original series, 
    #and add it to a cumulative sum of forecasted values. 
    #When working with second-order differences we first must perform this operation on the most recent first-order difference.
    
    # Add the most recent first difference from the training side of the original dataset to the forecast cumulative sum
    vectorAutoRegressiveMethodForecastedValues['Money1d'] = (vectorAutoRegressiveMethodDataset['Money'].iloc[-numberOfObsevation-1]-vectorAutoRegressiveMethodDataset['Money'].iloc[-numberOfObsevation-2]) + vectorAutoRegressiveMethodForecastedValues['Money2d'].cumsum()
    
    # Now build the forecast values from the first difference set
    vectorAutoRegressiveMethodForecastedValues['MoneyForecast'] = vectorAutoRegressiveMethodDataset['Money'].iloc[-numberOfObsevation-1] + vectorAutoRegressiveMethodForecastedValues['Money1d'].cumsum()
    
    # Add the most recent first difference from the training side of the original dataset to the forecast cumulative sum
    vectorAutoRegressiveMethodForecastedValues['Spending1d'] = (vectorAutoRegressiveMethodDataset['Spending'].iloc[-numberOfObsevation-1]-vectorAutoRegressiveMethodDataset['Spending'].iloc[-numberOfObsevation-2]) + vectorAutoRegressiveMethodForecastedValues['Spending2d'].cumsum()
    
    # Now build the forecast values from the first difference set
    vectorAutoRegressiveMethodForecastedValues['SpendingForecast'] = vectorAutoRegressiveMethodDataset['Spending'].iloc[-numberOfObsevation-1] + vectorAutoRegressiveMethodForecastedValues['Spending1d'].cumsum()
    
    #saving the foreasted values without lag
    saveVectorAutoRegressiveMethodForecastedValues(vectorAutoRegressiveMethodForecastedValues)

def varmaResultsPlot():
    
    #reading model from pickle file
    vectorAutoRegressiveMethodModel = readVectorAutoRegressiveMethodModel()
    
    visualizeResultPlots(vectorAutoRegressiveMethodModel)

def varmaForecastedsPlot():
    
    #reading model from pickle file
    vectorAutoRegressiveMethodModel = readVectorAutoRegressiveMethodModel()
    
    visualizeForecastedPlots(vectorAutoRegressiveMethodModel)
    
    

def plotVectorAutoRegressiveMethodPredictedValues():
    
    #reading the forecasted values
    vectorAutoRegressiveMethodForecastedValues = readVectorAutoRegressiveMethodForecastedValues()
    
    #reading the full dataset
    vectorAutoRegressiveMethodDataset = importVectorAutoRegressiveMethodDataset("M2SLMoneyStock.csv", "PCEPersonalSpending.csv")
    
    visualizeVectorAutoRegressiveMethodPredictedValuesForMoney(vectorAutoRegressiveMethodDataset, vectorAutoRegressiveMethodForecastedValues)
    
    visualizeVectorAutoRegressiveMethodPredictedValuesForSpending(vectorAutoRegressiveMethodDataset, vectorAutoRegressiveMethodForecastedValues)
    
if __name__ == "__main__":
    #varmaResultsPlot()
    #varmaForecastedsPlot()
    #testVectorAutoRegressiveMethodModel()
    plotVectorAutoRegressiveMethodPredictedValues()