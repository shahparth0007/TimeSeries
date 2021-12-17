## Outside libraries
import warnings
import streamlit as st
warnings.filterwarnings('ignore', 'statsmodels.tsa.arima_model.ARMA',
                        FutureWarning)
import pandas as pd
import os
import pandas as pd 
import numpy as np 
import itertools
from sklearn.metrics import mean_squared_error as mse
#Custom Functions
import config
import Forecast_01
import trainandtest
import transform
import AlgoArima
import HoltWinters
import AlgoProphet
import AlgoSARIMA

#Defining MAPE metric 
def MAPE(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true.sum() - y_pred.sum()) / y_true.sum())) * 100


#Customer Functions
def forecast(category,frequency = "W",level="ID",n=1,salesFromPath=True,
             flagsFromPath=True,transformation="",confidencelevel=0.95):
    sales, flags = Forecast_01.dataExtraction(category, level,salesFromPath,
                                                    flagsFromPath)
    unique=set(sales["ID"])
    algotable = pd.DataFrame()
    forecast = pd.Series()
    errortable_dict= {}

    def MAPE(y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true.sum() - y_pred.sum()) / y_true.sum())) * 100



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
        trainsales, testsales = trainandtest.traintestsplit(salesdata)
        trainflags, testflags = trainandtest.traintestsplit(flagsdata)

        ############################# MODEL TRAINING ############################# 
        #creating dataframe for storing mse for models 
        errortable = pd.DataFrame()


        ########## ARIMA ##########

        print("Starting ARIMA")

        try:
            #Transforming data to stationarize it 
            transformeddata = transform.transform(trainsales,confidencelevel,
                                                  transformation)
            #Passing scaled data to the arimax model 
            scaledforecast = AlgoArima.arimax(transformeddata[0],len(testsales))
            #Inverse scaling the scaled forecasts 
            arimaforecasts = transform.inverse(scaledforecast,transformeddata)
            error = mse(arimaforecasts ,testsales)
            error2 = MAPE(testsales,arimaforecasts)
            #Appending mse and MAPE to error table 
            errortable = errortable.append({'Algorithm' : 'ARIMA',
                                        "MAPE":error2},ignore_index=True)
            del scaledforecast, arimaforecasts
        except: 
            print("Arima__ An error occured while running Arima")
            errortable = errortable.append({'Algorithm':'ARIMA'},
                                                ignore_index=True)

        print("Starting ARIMAX")
        try:
            trainflags1 = trainflags.iloc[-len(transformeddata[0]):,:]
            transformeddata[0].index = trainflags1.index
            scaledforecast = AlgoArima.arimax(transformeddata[0],len(testsales),
                                         trainFlags=trainflags1,forecastFlags=testflags)
            arimaxforecasts = transform.inverse(scaledforecast,transformeddata)
            error = mse(arimaxforecasts ,testsales)
            error2 = MAPE(testsales,arimaxforecasts)
            errortable = errortable.append({'Algorithm' : 'ARIMAX',
                                            "MAPE":error2},ignore_index=True)
            del transformeddata, scaledforecast, arimaxforecasts, trainflags1
        except: 
            print("Arimax__ An error occured while running Arimax")
            errortable = errortable.append({'Algorithm':'ARIMAX'},
                                                ignore_index=True)  
            
               #only runs SARIMA/SARIMAX if data is at monthly level
        # print("Starting SARIMA")        
        # if frequency=="M":
        #     ########## SARIMA ##########    
        #     try:
        #         error, pred = AlgoSARIMA.sarimax(salesdata,len(testsales),f=frequency,only_error=True)
        #         error2 = MAPE(testsales,pred)
        #         errortable = errortable.append({'Algorithm':'SARIMA',
        #                                         "MAPE":error2},ignore_index=True) 
        #     except:
        #         print("Sarima__ An error occured while running Sarima")
        #         errortable = errortable.append({'Algorithm':'SARIMA','MAPE':np.nan
        #                                         },ignore_index=True) 
    
        #     ########## SARIMAX ##########
        #     try:
        #         error,pred = AlgoSARIMA.sarimax(salesdata,len(testsales),f=frequency,
        #                                            Flags=flagsdata, only_error=True)
        #         error2 = MAPE(testsales,pred)
        #         errortable = errortable.append({'Algorithm':'SARIMAX',
        #                                         "MAPE":error2},ignore_index=True)    
        #     except:
        #         print("Sarimax__ An error occured while running Sarimax")
        #         errortable = errortable.append({'Algorithm':'SARIMAX',
        #                                         'MAPE':np.nan},ignore_index=True)         
            
        # print("Starting Holt Winters")
        # try:
        #     holtMSE,algo = HoltWinters.HOLTWINTER(trainsales,testsales,category,
        #                                                 frequency)
        #     print("MAPE for HoltWinters:",holtMSE)
        #     if algo == "AANd":
        #         #AANd = Additive trend , Additive seasonality , No Damp
        #         errortable = errortable.append({'Algorithm' : 'HOLT-WINTERS-AANd', 
        #                                         "MAPE":holtMSE},ignore_index=True)                                     

        #     elif algo == "AMNd":    
        #         #AANd = Additive trend , Multiplicative seasonality , No Damp
        #         errortable = errortable.append({'Algorithm' : 'HOLT-WINTERS-AMNd', 
        #                                         "MAPE":holtMSE},ignore_index=True)                                 
        #     elif algo == "AAD":
        #         #AAD = Additive trend , Additive seasonality , With Damp
        #         errortable = errortable.append({'Algorithm' : 'HOLT-WINTERS-AAD', 
        #                                         "MAPE":holtMSE},ignore_index=True)                
        #     elif algo == "AMD":
        #         #AMD = Additive trend , Multiplicative seasonality , No Damp
        #         errortable = errortable.append({'Algorithm' : 'HOLT-WINTERS-AMD', 
        #                                         "MAPE":holtMSE},ignore_index=True)

        # except:
        #     print("HoltWinter__ An error occured while running HoltWinter")
        #     errortable = errortable.append({'Algorithm':'HOLT-WINTERS','MAPE':np.nan},
        #                                         ignore_index=True)  


        # ########### PROPHET WITHOUT FLAGS ###########
        # ### ADDITIVE SEASONALITY ###
        # print("Starting Prophet")
        # try:
        #     propForecast = AlgoProphet.prophet(trainsales,n=len(testsales),
        #                                    f=frequency)
        #     error = mse(propForecast,testsales)
        #     error2 = MAPE(testsales,propForecast)
        #     errortable = errortable.append({'Algorithm' : 'Prophet(seas-Add)', 
        #                                     "MAPE":error2},
        #                                    ignore_index=True)            
        # except:
        #     raise
        #     print("Prophet__ An error occured while running Prophet(seas-Add)")
        #     errortable = errortable.append({'Algorithm' : 'Prophet(seas-Add)', 
        #                                     'MAPE' : np.nan},ignore_index=True)               



        errortable.dropna(inplace=True)
        errortable_dict[unit] = errortable
        print(errortable_dict)
        algo = errortable.loc[(errortable['MAPE'].idxmin())]
        print("Best algorithm for forecasting ",unit," is ",algo.Algorithm, ", with error: ",algo.MAPE)
        algotable = algotable.append({"ID":unit,"Best Algorithm":algo.Algorithm,"MAPE":algo.MAPE},ignore_index=True)
        
    return algotable,errortable_dict

    


    




