
from statsmodels.tsa.api import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import warnings
import math
import config
warnings.filterwarnings("ignore")

#Peform the train and test split for timeseries data
#Category should be passed as "J" : Jewelery , "E" : Eyeware , "W" : Watches
# def train_test_split(df, split = 0.7):
    
#     testSize = round(0.7 * len(df))
#     #print(testSize)
    
#     trainInd = df1.index[:testSize]
#     train = df1.loc[trainInd,"Qty"]
#     train.index = range(0,testSize)
#     #print(train.head())

#     testInd = df1.index[testSize:]
#     test = df1.loc[testInd , "Qty"]
#     test.index = range(testSize,len(df))
#     #print(test.head())
    
#     return train,test

#Returns the seasonality according to the period of the data
#data_freq can be "D" , "W" , "M" and "Y"
#category can be "J" , "E" and "W" for Jewellery , Eyeware and Watches respectively
def seasonalityPeriod(category, data_freq = "D"):
    m = config.DICT_SEASONALITY[category][data_freq]
    return m

#Function that returns the exponential smoothing predictions with different combinations of parametes
def ExponentialSmoothPredictions(train ,test, period , seasonalParam , boolDamp):
    
    #Training the model on the train data
    model = ExponentialSmoothing(train+1 , seasonal_periods= period , trend="add" , seasonal = seasonalParam, damped = boolDamp).fit(use_boxcox=True)
    
    #Getting the predictions for the test data
    pred = model.predict(start=test.index[0], end=test.index[-1])
    
    #Returning the predictions
    return pred
    

#Holt winter algo implementation for forecasting
def HOLTWINTER(train, test ,category , data_freq = "D" , only_error = True, algoIndex = 0):
    #Get seasonality period
    m = seasonalityPeriod(category,data_freq)
    
    #Perform the train and test split
    #train,test = train_test_split(df)
    
    #Create an empty list to store the MSE for each for algo in Holt Winters
    mseLst = []        
    
    #Apply Holt Winters model
    dictHoltParams = { "seasonal" : ["add" , "mul","add", "mul"],
                       "damp" :     [False , False, True , True]
                       }
    
    #If the function is called with only_error set to false then dont calculate MSE and simply return the predictions for the forecasting period specified by the test dataset
    #Also, algorithm index is specified always when only_error is set to false
    if not only_error:
        return ExponentialSmoothPredictions(train,test, m ,dictHoltParams["seasonal"][algoIndex] , dictHoltParams["damp"][algoIndex] )
    
    #Case 1 : Additive trend and additive seasonality ; No Damp
    #Case 2 : Additive trend and multiplicative seasonality ; No Damp
    #Case 3 : Additive trend and additive seasonality ; With Damp
    #Case 4 : Additive trend and multiplicative seasonality ; With Damp
    for i in range(4):
        try:
            pred = ExponentialSmoothPredictions(train,test, m ,dictHoltParams["seasonal"][i] , dictHoltParams["damp"][i] )
            
            #Appending the MSE for the current algorithm to the list
            mseLst.append(mean_squared_error(test,pred))
            #print(np.sqrt(mseLst))
            
        except:
            #print("Error in Exponential Smoothing implementation.")
            mseLst.append(1000000000000000)
            #raise

    
    #Get the index of the algorithm with min MSE
    #mseLst = [100 if math.isnan(x) else x for x in mseLst]
    index = mseLst.index(min(mseLst))
    
    if index==0:
        #AANd = Additive trend , Additive seasonality , No Damp
        algo = "AANd"
    elif index == 1:
        #AANd = Additive trend , Multiplicative seasonality , No Damp
        algo = "AMNd"
    elif index ==2:
        #AAD = Additive trend , Additive seasonality , With Damp
        algo = "AAD"
    elif index ==3:
        #AMD = Additive trend , Multiplicative seasonality , No Damp
        algo = "AMD"
    
    #print(algo)
    
    return min(mseLst) , algo
