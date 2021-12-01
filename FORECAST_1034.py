
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error
import numpy as np 
import pandas as pd
from sklearn.metrics import mean_squared_error as MSE


#Computing optimal values of p and q for ARIMA parameters        
def pparameter(data):
    try:
        pacfvalues=pacf(data,nlags=45)
        CI = 1.96/(np.sqrt(len(data)))
        p=np.min(np.nonzero(abs(pacfvalues)<CI))-1
        return p
    
    except:
        raise
        print("arimaparameters__An error occured while computing optimal parameters for ARIMA")

def qparameter(data):
    try:
        acfvalues=acf(data,nlags=45)
        CI = 1.96/(np.sqrt(len(data)))
        try:
            q = np.min(np.nonzero(abs(acfvalues)<CI))-1
            return q
        except:
            print("arimaparameters__An error occured while computing optimal parameters for ARIMA")
    except:
        raise
        print("arimaparameters__An error occured while computing optimal parameters for ARIMA")
        
# function for forecasting values using ARIMA model 
def arimax(data,n,pval=None,qval=None,trainFlags=None,forecastFlags=None,
           only_error=False,testdata=None):  

    if pval!=None:
        p = pval
    else:
        p = pparameter(data)

    if qval!=None:
        q = qval 
    else:
        q = qparameter(data)
    model = ARIMA(data,order=(p,0,q),exog=trainFlags)
    results = model.fit(disp=0)
    forecast = results.predict(start=len(data),end=len(data)+n-1,exog=forecastFlags)
    if only_error==False:
        return forecast     
    else:
        error = mean_squared_error(forecast,testdata)
        return error
    
        
        

def TSLM(data,f="D",n=1,remove_trend=False,remove_seasonality=False,seasonality="additive",
         trainFlags=None,forecastFlags=None,only_error=False,testdata=None):
    """
    f- (seasonality): daily-"D" , weekly-"W", monthly-"M" 
    data should be a pandas series with sales data, flags a pandas dataframe with 
    exog var split into train and test 
    set remove_trend/remove_seasonality to True to run model after removing trend 
    and seasonality 
    n - no.of periods into future to forecast 
    """
    
    if f=="D":
        frequency = 365
    elif f=="W":
        frequency = 52
    elif f=="M":
        frequency = 12
    #Decomposing data into trend,seasonality and residual components
    result = seasonal_decompose(data,model=seasonality, freq=frequency,extrapolate_trend=True)
    
    # Removes trend component from data
    if remove_trend==True:
        if seasonality=="additive":
            data = data - result.trend 
        if seasonality=="multiplicative":
            data = data/result.trend 
            
    #Removes seasonality component from data
    if remove_seasonality==True:
        if seasonality=="additive":
            data = data - result.seasonal
        if seasonality=="multiplicative":
            data = data/result.seasonal
        
    #Calculates optimal value of p(hyperparameter) for fitting the AR model 
    pacfvalues=pacf(data,nlags=45)
    CI = 1.96/(np.sqrt(len(data)))
    paramp=np.min(np.nonzero(abs(pacfvalues)<CI))-1

    #Fits AR model and returns predicted future sales 
    forecast = arimax(data,n,pval=paramp,qval=0,trainFlags=trainFlags,forecastFlags=forecastFlags)
    
    #Creates line of best fit(on trend component) and extrapolates it for future trend
    #Adds appropriate Datatime index to future trend values and adds to forecasted 
    #residual values 
    if remove_trend==True:
        trend = np.polyfit(np.array(range(1,len(data)+1)),result.trend.values,1)
        extrapolateTrend = np.poly1d(trend)
        futureTrend = pd.Series(extrapolateTrend(range(len(data),len(data)+1+n)))
        futureTrend.index = pd.date_range(start=data.index[-1],periods=n+1,freq=f)
        if seasonality=="additive":
            forecast = forecast + futureTrend[1:] 
        if seasonality=="multiplicative":
            forecast = forecast * futureTrend[1:] 
    
    #Adds back seasonality component 
    if remove_seasonality==True:
        if frequency>n:
            futureSeasonal = pd.Series(result.seasonal.values[-frequency:-frequency+n+1])
        elif frequency<=n:
            futureSeasonalCycle = pd.Series(result.seasonal.values[-frequency:])
            count = int(n/frequency)
            futureSeasonal = futureSeasonalCycle
            for i in range(count):
                futureSeasonal = futureSeasonal.append(futureSeasonalCycle)
            futureSeasonal = futureSeasonal[:n+1]
        futureSeasonal.index = pd.date_range(start=data.index[-1],periods=n+1,freq=f)
        if seasonality=="additive":
            forecast = forecast + futureSeasonal[1:]
        if seasonality=="multiplicative":
            forecast = forecast * futureSeasonal[1:] 
        
        
    #forecast = np.round(forecast)
    if only_error==False:
        return forecast 
    
    else:
        error = mean_squared_error(forecast,testdata)
        return error
        

#Defining MAPE metric 
def MAPE(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


def allcombTSLM(trainsales,testsales,frequency,n,trainflags,testflags):
    
    #creating dataframe for storing MSE for models 
    errortable = pd.DataFrame()
    
    ############ TSLM WITHOUT FLAGS (MODEL - ADDITIVE) ############
    # AM - ADDITIVE MOEDL 
    ### RESIDUAL + TREND + SEASONALITY ###
    try:
        tslmForecast = TSLM(trainsales,f=frequency,n=len(testsales))
        error = MSE(tslmForecast,testsales)
        error2 = MAPE(testsales,tslmForecast)
        errortable = errortable.append({'Algorithm' : '(AM)TSLM', 'MSE' : error,
                                       "MAPE":error2},
                                       ignore_index=True)
    except:
        errortable = errortable.append({'Algorithm':'(AM)TSLM','MSE':np.nan},
                                       ignore_index=True)  
        
    ### RESIDUAL + SEASONALITY ###
    try:
        tslmForecast = TSLM(trainsales,f=frequency,n=len(testsales),
                                 remove_trend=True)
        error = MSE(tslmForecast,testsales)
        error2 = MAPE(testsales,tslmForecast)
        errortable = errortable.append({'Algorithm' : '(AM)TSLM(Trend)', 
                                        'MSE' : error,"MAPE":error2},
                                       ignore_index=True)            
    except:
        errortable = errortable.append({'Algorithm' : '(AM)TSLM(Trend)', 
                                        'MSE':np.nan},ignore_index=True)             
    
    ### RESIDUAL + TREND ###
    try:
        tslmForecast = TSLM(trainsales,f=frequency,n=len(testsales),
                                 remove_seasonality=True,)
        error = MSE(tslmForecast,testsales)
        error2 = MAPE(testsales,tslmForecast)
        errortable = errortable.append({'Algorithm' : '(AM)TSLM(Seas)', 
                                        'MSE' : error,"MAPE":error2},
                                       ignore_index=True)            
    except:
        errortable = errortable.append({'Algorithm' : '(AM)TSLM(Seas)', 
                                        'MSE':np.nan},ignore_index=True)               
    
    
    ### RESIDUAL ###
    try:
        tslmForecast = TSLM(trainsales,f=frequency,n=len(testsales),
                             remove_trend=True,remove_seasonality=True,)
        error = MSE(tslmForecast,testsales)
        error2 = MAPE(testsales,tslmForecast)
        errortable = errortable.append({'Algorithm' : '(AM)TSLM(Trend+Seas)', 
                                        'MSE' : error,"MAPE":error2},
                                       ignore_index=True)            
    except:
        errortable = errortable.append({'Algorithm' : '(AM)TSLM(Trend+Seas)', 
                                        'MSE':np.nan},ignore_index=True)         
    
    ############ TSLM WITH FLAGS (MODEL - ADDITIVE) ############
    ### RESIDUAL + TREND + SEASONALITY ###
    try:
        tslmForecast = TSLM(trainsales,f=frequency,n=len(testsales),
                                 trainFlags=trainflags,forecastFlags=testflags)
        error = MSE(tslmForecast,testsales)
        error2 = MAPE(testsales,tslmForecast)
        errortable = errortable.append({'Algorithm':'(AM)TSLMwFlags','MSE':error,
                                        "MAPE":error2}, ignore_index=True)
    except:
        errortable = errortable.append({'Algorithm' : '(AM)TSLMwFlags', 
                                        'MSE' : np.nan},ignore_index=True)            
    
    ### RESIDUAL + SEASONALITY ###
    try:
        tslmForecast = TSLM(trainsales,f=frequency,n=len(testsales),
                                 remove_trend=True,trainFlags=trainflags,
                                 forecastFlags=testflags)
        error = MSE(tslmForecast,testsales)
        error2 = MAPE(testsales,tslmForecast)
        errortable = errortable.append({'Algorithm' : '(AM)TSLMwFlags(Trend)', 
                                        'MSE' : error,"MAPE":error2},
                                       ignore_index=True)
    except:
        errortable = errortable.append({'Algorithm' : '(AM)TSLMwFlags(Trend)', 
                                        'MSE':np.nan},ignore_index=True)
    
    ### RESIDUAL + TREND ###
    try:
        tslmForecast = TSLM(trainsales,f=frequency,n=len(testsales),
                                 remove_seasonality=True,trainFlags=trainflags,
                                 forecastFlags=testflags)
        error = MSE(tslmForecast,testsales)
        error2 = MAPE(testsales,tslmForecast)
        errortable = errortable.append({'Algorithm' : '(AM)TSLMwFlags(Seas)',
                                        'MSE' : error,"MAPE":error2},
                                       ignore_index=True)
    except:
        errortable = errortable.append({'Algorithm' : '(AM)TSLMwFlags(Seas)', 
                                        'MSE':np.nan},ignore_index=True)            
    
    ### RESIDUAL ###
    try:
        tslmForecast = TSLM(trainsales,f=frequency,n=len(testsales),
                                 remove_trend=True,remove_seasonality=True,
                                 trainFlags=trainflags,forecastFlags=testflags)
        error = MSE(tslmForecast,testsales)
        error2 = MAPE(testsales,tslmForecast)
        errortable = errortable.append({'Algorithm':'(AM)TSLMwFlags(Trend+Seas)',
                                        'MSE' : error,"MAPE":error2},
                                       ignore_index=True)
    except:
        errortable = errortable.append({'Algorithm':'(AM)TSLMwFlags(Trend+Seas)', 
                                        'MSE':np.nan},ignore_index=True)             

    
    if not any(trainsales<1):                   
        # MM - MULTIPLICATIVE MODEL                
        ############ TSLM WITHOUT FLAGS (MODEL - MULTIPLICATIVE) ############  
        ### RESIDUAL * SEASONALITY ###
        try:
            tslmForecast = TSLM(trainsales,f=frequency,n=len(testsales),
                                     seasonality="multiplicative",remove_trend=True)
            error = MSE(tslmForecast,testsales)
            error2 = MAPE(testsales,tslmForecast)
            errortable = errortable.append({'Algorithm' : '(MM)TSLM(Trend)', 
                                            'MSE' : error,"MAPE":error2},
                                           ignore_index=True)            
        except:
            errortable = errortable.append({'Algorithm' : 'TSLM(Trend)', 
                                            'MSE':np.nan},ignore_index=True)             
        
        ### RESIDUAL * TREND ###
        try:
            tslmForecast = TSLM(trainsales,f=frequency,remove_seasonality=True,
                                     seasonality="multiplicative",n=len(testsales))
            error = MSE(tslmForecast,testsales)
            error2 = MAPE(testsales,tslmForecast)
            errortable = errortable.append({'Algorithm' : '(MM)TSLM(Seas)', 
                                            'MSE':error,"MAPE":error2},
                                           ignore_index=True)            
        except:
            errortable = errortable.append({'Algorithm' : '(MM)TSLM(Seas)', 
                                            'MSE':np.nan},ignore_index=True)               
        
        
        ### RESIDUAL ###
        try:
            tslmForecast = TSLM(trainsales,f=frequency,n=len(testsales),
                                     remove_trend=True,remove_seasonality=True,
                                     seasonality="multiplicative")
            error = MSE(tslmForecast,testsales)
            error2 = MAPE(testsales,tslmForecast)
            errortable = errortable.append({'Algorithm' : '(MM)TSLM(Trend+Seas)', 
                                            'MSE' : error,"MAPE":error2},
                                           ignore_index=True)            
        except:
            errortable = errortable.append({'Algorithm' : '(MM)TSLM(Trend+Seas)', 
                                            'MSE':np.nan},ignore_index=True)
        
        ############ TSLM WITH FLAGS (MODEL - MULTIPLICATIVE) ###########       
        ### RESIDUAL * SEASONALITY ###
        try:
            tslmForecast = TSLM(trainsales,f=frequency,remove_trend=True,
                                     trainFlags=trainflags,seasonality="multiplicative",
                                     n=len(testsales),forecastFlags=testflags)
            error = MSE(tslmForecast,testsales)
            error2 = MAPE(testsales,tslmForecast)
            errortable = errortable.append({'Algorithm' : '(MM)TSLMwFlags(Trend)', 
                                            'MSE' : error,"MAPE":error2},
                                           ignore_index=True)
        except:
            errortable = errortable.append({'Algorithm' : '(MM)TSLMwFlags(Trend)', 
                                            'MSE' : np.nan},ignore_index=True)
        
        ### RESIDUAL * TREND ###
        try:
            tslmForecast = TSLM(trainsales,f=frequency,remove_seasonality=True,
                                     seasonality="multiplicative",n=len(testsales),
                                     trainFlags=trainflags,forecastFlags=testflags)
            error = MSE(tslmForecast,testsales)
            error2 = MAPE(testsales,tslmForecast)
            errortable = errortable.append({'Algorithm' : '(MM)TSLMwFlags(Seas)', 
                                            'MSE' : error,"MAPE":error2},
                                           ignore_index=True)
        except:
            errortable = errortable.append({'Algorithm' : '(MM)TSLMwFlags(Seas)', 
                                            'MSE' : np.nan},ignore_index=True)            
        
        ### RESIDUAL ###
        try:
            tslmForecast = TSLM(trainsales,f=frequency,n=len(testsales),
                                     remove_trend=True,trainFlags=trainflags,
                                     remove_seasonality=True,forecastFlags=testflags,
                                     seasonality="multiplicative")
            error = MSE(tslmForecast,testsales)
            error2 = MAPE(testsales,tslmForecast)
            errortable = errortable.append({'Algorithm' : '(MM)TSLMwFlags(Trend+Seas)', 
                                            'MSE' : error,"MAPE":error2},
                                           ignore_index=True)
        except:
            errortable = errortable.append({'Algorithm' : '(MM)TSLMwFlags(Trend+Seas)', 
                                            'MSE' : np.nan},ignore_index=True)
        
        #Extracting algorithm name and MSE,MAPE of best algorithm 
    row = errortable.loc[errortable['MSE'].idxmin()]
    algorithm    = row["Algorithm"]
    mse = row["MSE"]
    mape = row["MAPE"]
    
    return algorithm,mse,mape
         
          
