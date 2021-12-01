# -*- coding: utf-8 -*-
"""
"""
import os
import pandas as pd 
import numpy as np 
from sklearn.metrics import mean_squared_error as mse
import FORECAST_1001 
import FORECAST_1031 
import transform
import FORECAST_1036
import FORECAST_1034
import traintest
import FORECAST_1035
import FORECAST_1004

#Defining MAPE metric 
def MAPE(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true.sum() - y_pred.sum()) / y_true.sum())) * 100

def forecast(category,level="ID",frequency="M",n=1,salesFromPath=True,
             flagsFromPath=True,transformation="",confidencelevel=0.95):

    ############################# DATA EXTRACTION #############################
    sales, flags = FORECAST_1001.dataExtraction(category, level,salesFromPath,
                                                flagsFromPath)
    unique=set(sales["ID"])
    algotable = pd.DataFrame()
    forecast = pd.Series()
    ############################## DATA PRE-PROCESSING ########################
    #Looping over every individual item in col "ID"
    for unit in unique:
        #Subsetting sales and flags data for each unique item in "ID" column of
        #salesdata df for training and forecasting 
        print("Starting forecast for CATPB: ",unit)
        salesdata = pd.Series(sales.loc[sales["ID"]==unit,"SALES"])
        startdate = salesdata.index[0]
        #startdate = salesdata[np.min(salesdata):].index[0]
        salesdata = salesdata[startdate:]
        flagsdata = flags[startdate:]  

        #Grouping data to a weekly level for training/forecasting
        if frequency=="W":
            salesdata = salesdata.resample('W').sum()
            flagsdata = flagsdata.drop('WEEKEND_FLAG',axis=1,errors='ignore')
            flagsdata = flagsdata.resample('W').mean()
            
        #Grouping data to a monthly level for training/forecasting
        if frequency=="M":
            salesdata = salesdata.resample('M').sum()
            flagsdata = flagsdata.drop(["Start of Month Flag","MONTHEND_FLAG",
                                        "End of Month Flag"],axis=1,errors='ignore')
            flagsdata = flagsdata.resample('M').mean()        

        flagsdf = flagsdata.iloc[len(salesdata):len(salesdata)+n]   
        flagsdata = flagsdata.iloc[:len(salesdata)]        
        
        ######### TRAIN-TEST SPLIT #########
        trainsales, testsales = traintest.traintestsplit(salesdata)
        trainflags, testflags = traintest.traintestsplit(flagsdata)
        
        ############################# MODEL TRAINING ############################# 
        #creating dataframe for storing mse for models 
        errortable = pd.DataFrame()
        
        ########## ARIMA ##########
        try:
            #Transforming data to stationarize it 
            transformeddata = transform.transform(trainsales,confidencelevel,
                                                  transformation)
            #Passing scaled data to the arimax model 
            scaledforecast = FORECAST_1034.arimax(transformeddata[0],len(testsales))
            #Inverse scaling the scaled forecasts 
            arimaforecasts = transform.inverse(scaledforecast,transformeddata)
            error = mse(arimaforecasts ,testsales)
            error2 = MAPE(testsales,arimaforecasts)
            #Appending mse and MAPE to error table 
            errortable = errortable.append({'Algorithm' : 'ARIMA', 'mse' : error, 
                                            "MAPE":error2},ignore_index=True)            
            del scaledforecast, arimaforecasts
        
        except:
            print("Arima__ An error occured while running Arima")
            errortable = errortable.append({'Algorithm':'ARIMA','mse':np.nan},
                                            ignore_index=True)                     
        
        ########## ARIMAX ##########
        try:
            trainflags1 = trainflags.iloc[-len(transformeddata[0]):,:]
            transformeddata[0].index = trainflags1.index
            scaledforecast = FORECAST_1034.arimax(transformeddata[0],len(testsales),
                                         trainFlags=trainflags1,forecastFlags=testflags)
            arimaxforecasts = transform.inverse(scaledforecast,transformeddata)
            error = mse(arimaxforecasts ,testsales)
            error2 = MAPE(testsales,arimaxforecasts)
            errortable = errortable.append({'Algorithm' : 'ARIMAX', 'mse' : error,
                                            "MAPE":error2},ignore_index=True)
            del transformeddata, scaledforecast, arimaxforecasts, trainflags1
        
        except:
            print("Arimax__ An error occured while running Arimax")
            errortable = errortable.append({'Algorithm':'ARIMAX','mse':np.nan},
                                            ignore_index=True)            
        
        #only runs SARIMA/SARIMAX if data is at monthly level
        if frequency=="M":
            ########## SARIMA ##########    
            try:
                error, pred = FORECAST_1035.sarimax(salesdata,len(testsales),f=frequency,only_error=True)
                error2 = MAPE(testsales,pred)
                errortable = errortable.append({'Algorithm':'SARIMA','mse':error,
                                                "MAPE":error2},ignore_index=True) 
            except:
                print("Sarima__ An error occured while running Sarima")
                errortable = errortable.append({'Algorithm':'SARIMA',
                                                'mse':np.nan},ignore_index=True) 
    
            ########## SARIMAX ##########
            try:
                error,pred = FORECAST_1035.sarimax(salesdata,len(testsales),f=frequency,
                                                   Flags=flagsdata, only_error=True)
                error2 = MAPE(testsales,pred)
                errortable = errortable.append({'Algorithm':'SARIMAX','mse':error,
                                                "MAPE":error2},ignore_index=True)    
            except:
                print("Sarimax__ An error occured while running Sarimax")
                errortable = errortable.append({'Algorithm':'SARIMAX',
                                                'mse':np.nan},ignore_index=True)    
        
        ######### HOLT-WINTERS #########
        holtMSE,algo = FORECAST_1031.HOLTWINTER(trainsales,testsales,category,
                                                frequency)
        if algo == "AANd":
            #AANd = Additive trend , Additive seasonality , No Damp
            errortable = errortable.append({'Algorithm' : 'HOLT-WINTERS-AANd', 
                                            'mse' : holtMSE},ignore_index=True)                                     
            
        elif algo == "AMNd":    
            #AANd = Additive trend , Multiplicative seasonality , No Damp
            errortable = errortable.append({'Algorithm' : 'HOLT-WINTERS-AMNd', 
                                            'mse' : holtMSE},ignore_index=True)                                 
        elif algo == "AAD":
            #AAD = Additive trend , Additive seasonality , With Damp
            errortable = errortable.append({'Algorithm' : 'HOLT-WINTERS-AAD', 
                                            'mse' : holtMSE},ignore_index=True)                                          
        elif algo == "AMD":
            #AMD = Additive trend , Multiplicative seasonality , No Damp
            errortable = errortable.append({'Algorithm' : 'HOLT-WINTERS-AMD', 
                                            'mse' : holtMSE},ignore_index=True)                                    
        
        
        ######### TSLM #########     
        alg,MSE,mape = FORECAST_1034.allcombTSLM(trainsales,testsales,frequency,n,trainflags,testflags)
        
        errortable = errortable.append({'Algorithm' : alg, 'mse' : MSE,  
                                        'MAPE' : mape},ignore_index=True)               
            
        ########### PROPHET WITHOUT FLAGS ###########
        ### ADDITIVE SEASONALITY ###
        try:
            propForecast = FORECAST_1036.prophet(trainsales,n=len(testsales),
                                           f=frequency)
            error = mse(propForecast,testsales)
            error2 = MAPE(testsales,propForecast)
            errortable = errortable.append({'Algorithm' : 'Prophet(seas-Add)', 
                                            'mse' : error,"MAPE":error2},
                                           ignore_index=True)            
        except:
            raise
            print("Prophet__ An error occured while running Prophet(seas-Add)")
            errortable = errortable.append({'Algorithm' : 'Prophet(seas-Add)', 
                                            'mse' : np.nan},ignore_index=True)               
            
        ### MULTIPLICATIVE SEASONALITY ###
        try:
            propForecast = FORECAST_1036.prophet(trainsales,n=len(testsales),f=frequency,
                                                 seasonality='multiplicative')
            error = mse(propForecast,testsales)
            error2 = MAPE(testsales,propForecast)
            errortable = errortable.append({'Algorithm' : 'Prophet(seas-Mult)', 
                                            'mse' : error,"MAPE":error2},
                                           ignore_index=True)            
        except:
            print("Prophet__ An error occured while running Prophet(seas-Mult)")
            errortable = errortable.append({'Algorithm' : 'Prophet(seas-Mult)', 
                                            'mse' : np.nan},ignore_index=True)            
    
        
        ############# PROPHET WITH FLAGS #############
        ### ADDITIVE SEASONALITY ###
        try:
            propForecast = FORECAST_1036.prophet(trainsales,n=len(testsales),
                                                 f=frequency,flags=flagsdata)
            error = mse(propForecast,testsales)
            error2 = MAPE(testsales,propForecast)
            errortable = errortable.append({'Algorithm' : 'ProphetwFlags(seas-Add)', 
                                            'mse' : error,"MAPE":error2},
                                           ignore_index=True)            
        except:
            print("Prophet__ An error occured while running ProphetwFlags(seas-Add)")
            errortable = errortable.append({'Algorithm' : 'ProphetwFlags(seas-Add)', 
                                            'mse' : np.nan},ignore_index=True)              
        
        ### MULTIPLICATIVE SEASONALITY ###
        try:
            propForecast = FORECAST_1036.prophet(trainsales,n=len(testsales),flags=flagsdata,
                                                 seasonality='multiplicative',f=frequency)
            error = mse(propForecast,testsales)
            error2 = MAPE(testsales,propForecast)
            errortable = errortable.append({'Algorithm' : 'ProphetwFlags(seas-Mult)', 
                                            'mse' : error,"MAPE":error2},
                                           ignore_index=True)    
        except:
            print("Prophet__ An error occured while running ProphetwFlags(seas-Mult)")
            errortable = errortable.append({'Algorithm' : 'ProphetwFlags(seas-Mult)', 
                                            'mse' : np.nan},ignore_index=True)       
        del propForecast
        print(errortable)
        
        """
        ######### LSTM #########
        if error > config.MSE_THRESHOLD:
            #Go for LSTM algorithm
        else:
            #Do not go for LSTM
        """    


        ############################### FORECASTING ###############################
        #Selecting algorithm with least mse to forecast for future 
        errortable.dropna(inplace=True)
        algo = errortable.loc[(errortable['mse'].idxmin())]
        print("Best algorithm for forecasting ",unit," is ",algo.Algorithm, ", with error: ",algo.mse)
        algotable = algotable.append({"ID":unit,"algorithm":algo.Algorithm,"mse":algo.mse},ignore_index=True)
        #if algo=="LSTM":  
        unitname = pd.Series(unit)
        forecast1 = FORECAST_1004.prediction(n,frequency,algo.Algorithm,salesdata,flagsdata,
                                            flagsdf,category,trainflags,testflags,
                                            transformation,confidencelevel,testsales)
        forecast = forecast.append([unitname,forecast1])  
        
        ################################ EXPORTING ################################
    forecast.to_csv("prediction"+".csv") 
    algotable.to_csv("algorithm"+".csv") 

forecast_1 = forecast(category = 'J', n=3)
forecast_1