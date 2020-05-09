# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:53:28 2020

@author: Santosh Sah
"""
import pylab

def visualizeVectorAutoRegressiveMethodPredictedValuesForMoney(vectorAutoRegressiveMethodDataset, vectorAutoRegressiveMethodForecastedValues):
    
    numberOfObservation = 12
    
    #plotting the predicted values
    vectorAutoRegressiveMethodDataset['Money'][-numberOfObservation:].plot(figsize=(12,5),legend=True).autoscale(axis='x',tight=True)
    vectorAutoRegressiveMethodForecastedValues['MoneyForecast'].plot(legend=True);
    
    pylab.savefig('PredeictedValuesForMoney.png')

def visualizeVectorAutoRegressiveMethodPredictedValuesForSpending(vectorAutoRegressiveMethodDataset, vectorAutoRegressiveMethodForecastedValues):
    
    numberOfObservation = 12
    
    #plotting the predicted values
    vectorAutoRegressiveMethodDataset['Spending'][-numberOfObservation:].plot(figsize=(12,5),legend=True).autoscale(axis='x',tight=True)
    vectorAutoRegressiveMethodForecastedValues['SpendingForecast'].plot(legend=True);
    
    pylab.savefig('PredeictedValuesForSpending.png')

def visualizeVectorAutoRegressiveMethodForecastedValues(vectorAutoRegressiveMethodDataset, vectorAutoRegressiveMethodForecastedValues):
    
    #plotting the forecated values with full dataset
    vectorAutoRegressiveMethodDataset["PopEst"].plot()
    
    vectorAutoRegressiveMethodForecastedValues.plot()
    
    pylab.savefig('ForecastedValues.png')

def visualizeSourceDataPlot(vectorAutoRegressiveMethodDataset):
    
    #plotting the source dataset
    title = 'M2 Money Stock vs. Personal Consumption Expenditures'
    
    ylabel='Billions of dollars'
    
    xlabel='' 

    ax = vectorAutoRegressiveMethodDataset['Spending'].plot(figsize=(16,5),title=title, legend = True)
    
    ax.autoscale(axis='x',tight=True)
    
    ax.set(xlabel=xlabel, ylabel=ylabel)
    
    vectorAutoRegressiveMethodDataset['Money'].plot(legend=True)
    
    pylab.savefig('SourceDatasetPlot.png')

def visualizeResultPlots(vectorAutoRegressiveMethodModel):
    
    vectorAutoRegressiveMethodModel.plot()
    
    pylab.savefig('VARMAResultsPlot.png')

def visualizeForecastedPlots(vectorAutoRegressiveMethodModel):
    
    vectorAutoRegressiveMethodModel.plot_forecast(12)
    
    pylab.savefig('VARMAForecastedPlot.png')