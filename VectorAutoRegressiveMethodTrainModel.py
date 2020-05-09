# -*- coding: utf-8 -*-
"""
Created on Tue May  5 11:51:54 2020

@author: Santosh Sah
"""
from statsmodels.tsa.api import VAR

from VectorAutoRegressiveMethodUtils import (saveVectorAutoRegressiveMethodModel, readVectorAutoRegressiveMethodXTrain, 
                                               importVectorAutoRegressiveMethodDataset, saveVectorAutoRegressiveMethodModelForFullDataset)

from VectorAutoRegressiveMethodVisualization import (visualizeSourceDataPlot)

"""
Train VectorAutoRegressiveMethod model on training set
"""
def trainVectorAutoRegressiveMethodModel():
    
    X_train = readVectorAutoRegressiveMethodXTrain()
    
    #training model on the training set
    vectorAutoRegressiveMethodModel = VAR(X_train)  
    
    #we are taking p = 5 as we have created different models based on the different p values.
    #Model gives minimum aic and bic for p =5
    vectorAutoRegressiveMethodModelResult = vectorAutoRegressiveMethodModel.fit(5)
    
    #saving the model in pickle file
    saveVectorAutoRegressiveMethodModel(vectorAutoRegressiveMethodModelResult)
    
    print(vectorAutoRegressiveMethodModelResult.summary())
    
# =============================================================================
#      Summary of Regression Results
#     ==================================
#     Model:                         VAR
#     Method:                        OLS
#     Date:           Sat, 09, May, 2020
#     Time:                     12:54:09
#     --------------------------------------------------------------------
#     No. of Equations:         2.00000    BIC:                    14.1131
#     Nobs:                     233.000    HQIC:                   13.9187
#     Log likelihood:          -2245.45    FPE:                    972321.
#     AIC:                      13.7873    Det(Omega_mle):         886628.
#     --------------------------------------------------------------------
#     Results for equation Money
#     ==============================================================================
#                      coefficient       std. error           t-stat            prob
#     ------------------------------------------------------------------------------
#     const               0.516683         1.782238            0.290           0.772
#     L1.Money           -0.646232         0.068177           -9.479           0.000
#     L1.Spending        -0.107411         0.051388           -2.090           0.037
#     L2.Money           -0.497482         0.077749           -6.399           0.000
#     L2.Spending        -0.192202         0.068613           -2.801           0.005
#     L3.Money           -0.234442         0.081004           -2.894           0.004
#     L3.Spending        -0.178099         0.074288           -2.397           0.017
#     L4.Money           -0.295531         0.075294           -3.925           0.000
#     L4.Spending        -0.035564         0.069664           -0.511           0.610
#     L5.Money           -0.162399         0.066700           -2.435           0.015
#     L5.Spending        -0.058449         0.051357           -1.138           0.255
#     ==============================================================================
#     
#     Results for equation Spending
#     ==============================================================================
#                      coefficient       std. error           t-stat            prob
#     ------------------------------------------------------------------------------
#     const               0.203469         2.355446            0.086           0.931
#     L1.Money            0.188105         0.090104            2.088           0.037
#     L1.Spending        -0.878970         0.067916          -12.942           0.000
#     L2.Money            0.053017         0.102755            0.516           0.606
#     L2.Spending        -0.625313         0.090681           -6.896           0.000
#     L3.Money           -0.022172         0.107057           -0.207           0.836
#     L3.Spending        -0.389041         0.098180           -3.963           0.000
#     L4.Money           -0.170456         0.099510           -1.713           0.087
#     L4.Spending        -0.245435         0.092069           -2.666           0.008
#     L5.Money           -0.083165         0.088153           -0.943           0.345
#     L5.Spending        -0.181699         0.067874           -2.677           0.007
#     ==============================================================================
#     
#     Correlation matrix of residuals
#                    Money  Spending
#     Money       1.000000 -0.267934
#     Spending   -0.267934  1.000000
# =============================================================================


"""
Train VectorAutoRegressiveMethod model on full dataset
"""
def trainVectorAutoRegressiveMethodModelOnFullDataset():
    
    vectorAutoRegressiveMethodDataset = importVectorAutoRegressiveMethodDataset("M2SLMoneyStock.csv", "PCEPersonalSpending.csv")
    
    #training model on the whole dataset
    vectorAutoRegressiveMethodModel = VAR(vectorAutoRegressiveMethodDataset)  
    
    #we are taking p = 5 as we have created different models based on the different p values.
    #Model gives minimum aic and bic for p =5
    vectorAutoRegressiveMethodModelResult = vectorAutoRegressiveMethodModel.fit(5)
    
    #saving the model in pickle files
    saveVectorAutoRegressiveMethodModelForFullDataset(vectorAutoRegressiveMethodModelResult)
    
    print(vectorAutoRegressiveMethodModelResult.summary())
    
# =============================================================================
#      Summary of Regression Results
#     ==================================
#     Model:                         VAR
#     Method:                        OLS
#     Date:           Sat, 09, May, 2020
#     Time:                     13:24:02
#     --------------------------------------------------------------------
#     No. of Equations:         2.00000    BIC:                    13.8778
#     Nobs:                     247.000    HQIC:                   13.6911
#     Log likelihood:          -2354.26    FPE:                    778666.
#     AIC:                      13.5652    Det(Omega_mle):         713683.
#     --------------------------------------------------------------------
#     Results for equation Money
#     ==============================================================================
#                      coefficient       std. error           t-stat            prob
#     ------------------------------------------------------------------------------
#     const              -3.578575        10.765839           -0.332           0.740
#     L1.Money            1.166264         0.064984           17.947           0.000
#     L1.Spending        -0.128001         0.049702           -2.575           0.010
#     L2.Money           -0.093662         0.100549           -0.931           0.352
#     L2.Spending         0.013760         0.070046            0.196           0.844
#     L3.Money            0.073985         0.102312            0.723           0.470
#     L3.Spending         0.096473         0.070343            1.371           0.170
#     L4.Money           -0.259604         0.101794           -2.550           0.011
#     L4.Spending         0.152404         0.069032            2.208           0.027
#     L5.Money            0.114124         0.064564            1.768           0.077
#     L5.Spending        -0.131595         0.049884           -2.638           0.008
#     ==============================================================================
#     
#     Results for equation Spending
#     ==============================================================================
#                      coefficient       std. error           t-stat            prob
#     ------------------------------------------------------------------------------
#     const              32.631897        14.361180            2.272           0.023
#     L1.Money            0.167879         0.086685            1.937           0.053
#     L1.Spending         1.009837         0.066300           15.231           0.000
#     L2.Money           -0.241628         0.134129           -1.801           0.072
#     L2.Spending         0.121418         0.093438            1.299           0.194
#     L3.Money            0.026879         0.136479            0.197           0.844
#     L3.Spending        -0.014393         0.093835           -0.153           0.878
#     L4.Money           -0.048322         0.135789           -0.356           0.722
#     L4.Spending        -0.067379         0.092086           -0.732           0.464
#     L5.Money            0.100564         0.086126            1.168           0.243
#     L5.Spending        -0.055003         0.066543           -0.827           0.408
#     ==============================================================================
# 
#     Correlation matrix of residuals
#                    Money  Spending
#     Money       1.000000 -0.239601
#     Spending   -0.239601  1.000000
# =============================================================================


def plotTheSourceData():
    
    vectorAutoRegressiveMethodDataset = importVectorAutoRegressiveMethodDataset("M2SLMoneyStock.csv", "PCEPersonalSpending.csv")
    
    visualizeSourceDataPlot(vectorAutoRegressiveMethodDataset)

def determineOrderOfP():
    
    X_train = readVectorAutoRegressiveMethodXTrain()
    
    for i in [1, 2, 3, 4, 5, 6, 7]:
        
        vectorAutoRegressiveMethodModel = VAR(X_train)
        
        vectorAutoRegressiveMethodModelResult = vectorAutoRegressiveMethodModel.fit(i)
        
        print('Order =', i)
        
        print('AIC: ', vectorAutoRegressiveMethodModelResult.aic)
        
        print('BIC: ', vectorAutoRegressiveMethodModelResult.bic)
        
        print()
        
        #var(5) return the lowest aic and bic values. Hence we will be building our model with p = 5
        
# =============================================================================
#         Order = 1
#         AIC:  14.178610495220896
#         BIC:  14.266409486135709
#         
#         Order = 2
#         AIC:  13.955189367163705
#         BIC:  14.101961901274958
#         
#         Order = 3
#         AIC:  13.849518291541038
#         BIC:  14.055621258341116
#         
#         Order = 4
#         AIC:  13.827950574458283
#         BIC:  14.093744506408877
#         
#         Order = 5
#         AIC:  13.78730034460964
#         BIC:  14.113149468980652
#         
#         Order = 6
#         AIC:  13.799076756885809
#         BIC:  14.185349048538068
#         
#         Order = 7
#         AIC:  13.797638727913972
#         BIC:  14.244705963046671
# =============================================================================
        

if __name__ == "__main__":
    #plotTheSourceData()
    #determineOrderOfP()
    #trainVectorAutoRegressiveMethodModel()
    trainVectorAutoRegressiveMethodModelOnFullDataset()    
