import pandas as pd 
import FORECAST_1031 
import transform
import FORECAST_1036
import FORECAST_1034
import FORECAST_1035


def prediction(n,frequency,algo,salesdata,flagsdata,flagsdf,category,trainflags,
                  testflags,transformation,confidencelevel,testsales):

    if "ARIMA" in algo:
        
        if algo=="ARIMA":
            transformeddata = transform.transform(salesdata,confidencelevel,
                                                  transformation)
            scaledforecast = FORECAST_1034.arimax(transformeddata[0],n)
            forecast = pd.Series(transform.inverse(scaledforecast,transformeddata))
            forecast.index = pd.date_range(start=salesdata.index[-1], 
                                           periods=n+1, freq=frequency)[1:]
            
        elif algo=="SARIMA":
            forecast = pd.Series(FORECAST_1035.sarimax(salesdata,len(testsales),f=frequency,
                                                 only_error=False,n=n))
            forecast.index = pd.date_range(start=salesdata.index[-1], 
                                           periods=n+1, freq=frequency)[1:]
            
        elif algo=="ARIMAX":        
            transformeddata = transform.transform(salesdata,confidencelevel,
                                                  transformation)
            flags1 = flagsdata.iloc[-len(transformeddata[0]):,:]
            transformeddata[0].index = flags1.index
            scaledforecast = FORECAST_1034.arimax(transformeddata[0],n,trainFlags=flagsdata,
                                         forecastFlags=flagsdf)
            forecast = pd.Series(transform.inverse(scaledforecast,transformeddata))
            forecast.index = pd.date_range(start=salesdata.index[-1], 
                                           periods=n+1, freq=frequency)[1:]
            
        elif algo=="SARIMAX":        
            forecast = pd.Series(FORECAST_1035.sarimax(salesdata,len(testsales),n=n,
                                                 f=frequency,only_error=False,
                                                 Flags=flagsdata,exog=flagsdf))       
            forecast.index = pd.date_range(start=salesdata.index[-1], 
                                           periods=n+1, freq=frequency)[1:]
        
        
    elif "HOLT" in algo:
        forecast = pd.Series([0 for i in range(n)])
        forecast.index = pd.date_range(start=salesdata.index[-1], 
                                       periods=n+1, freq=frequency)[1:]
        if algo=="HOLT-WINTERS-AANd":
            forecast = FORECAST_1031.HOLTWINTER(salesdata,forecast,category,
                                            frequency, False, 0)
        elif algo=="HOLT-WINTERS-AMNd":           
            forecast = FORECAST_1031.HOLTWINTER(salesdata,forecast,category,
                                            frequency, False, 1)
        elif algo=="HOLT-WINTERS-AAD":         
            forecast = FORECAST_1031.HOLTWINTER(salesdata,forecast,category,
                                            frequency, False, 2)
        elif algo=="HOLT-WINTERS-AMD": 
            forecast = FORECAST_1031.HOLTWINTER(salesdata,forecast,category,
                                            frequency, False, 3)
   
    elif "TSLM" in algo:      
        
        if algo=="(AM)TSLM":  
            forecast = FORECAST_1034.TSLM(salesdata,f=frequency,n=n)
        
        elif algo=="(AM)TSLM(Trend)":             
            forecast = FORECAST_1034.TSLM(salesdata,f=frequency,n=n,remove_trend=True)            
        
        elif algo=="(AM)TSLM(Seas)":             
            forecast = FORECAST_1034.TSLM(salesdata,f=frequency,n=n,remove_seasonality=True)            
        
        elif algo=="(AM)TSLM(Trend+Seas)":             
            forecast = FORECAST_1034.TSLM(salesdata,f=frequency,n=n,remove_trend=True,
                                 remove_seasonality=True)        
            
        elif algo=="(AM)TSLMwFlags":             
            forecast = FORECAST_1034.TSLM(salesdata,f=frequency,n=n,trainFlags=flagsdata,
                                 forecastFlags=flagsdf)          
            
        elif algo=="(AM)TSLMwFlags(Trend)":             
            forecast = FORECAST_1034.TSLM(salesdata,f=frequency,n=n,remove_trend=True,
                                 trainFlags=flagsdata,forecastFlags=flagsdf)           
            
        elif algo=="(AM)TSLMwFlags(Seas)":             
            forecast = FORECAST_1034.TSLM(salesdata,f=frequency,n=n,remove_seasonality=True,
                                 trainFlags=flagsdata,forecastFlags=flagsdf)            
            
        elif algo=="(AM)TSLMwFlags(Trend+Seas)":            
            forecast = FORECAST_1034.TSLM(salesdata,f=frequency,n=n,remove_trend=True,
                                 remove_seasonality=True,trainFlags=flagsdata,
                                 forecastFlags=flagsdf)           
            
        elif algo=="(MM)TSLM(Trend)":            
            forecast = FORECAST_1034.TSLM(salesdata,f=frequency,n=n,remove_trend=True,
                                 seasonality="multiplicative")           
            
        elif algo=="(MM)TSLM(Seas)":            
            forecast = FORECAST_1034.TSLM(salesdata,f=frequency,remove_seasonality=True,
                                 seasonality="multiplicative",n=n)            
            
        elif algo=="(MM)TSLM(Trend+Seas)":            
            forecast = FORECAST_1034.TSLM(salesdata,f=frequency,n=n,
                                 remove_trend=True,remove_seasonality=True,
                                 seasonality="multiplicative")            
            
        elif algo=="(MM)TSLMwFlags(Trend)":            
            forecast = FORECAST_1034.TSLM(salesdata,f=frequency,remove_trend=True,
                                 trainFlags=flagsdata,seasonality="multiplicative",
                                 n=n,forecastFlags=flagsdf)           
            
        elif algo=="(MM)TSLMwFlags(Seas)":            
            forecast = FORECAST_1034.TSLM(salesdata,f=frequency,remove_seasonality=True,
                                 seasonality="multiplicative",n=n,
                                 trainFlags=flagsdata,forecastFlags=flagsdf)            
            
        elif algo=="(MM)TSLMwFlags(Trend+Seas)":            
            forecast = FORECAST_1034.TSLM(salesdata,f=frequency,n=n,
                                 remove_trend=True,trainFlags=flagsdata,
                                 remove_seasonality=True,forecastFlags=flagsdf,
                                 seasonality="multiplicative")     
        
            
    elif "Prophet" in algo:    
        
        if algo=="Prophet(seas-Add)":            
            forecast = FORECAST_1036.prophet(salesdata,n=n,f=frequency)            
            
        elif algo=="Prophet(seas-Mult)":            
            forecast = FORECAST_1036.prophet(salesdata,n=n,f=frequency,
                                       seasonality='multiplicative')            
            
        elif algo=="ProphetwFlags(seas-Add)":            
            forecast = FORECAST_1036.prophet(salesdata,n=n,flags=flagsdata,
                                       f=frequency)            
            
        elif algo=="ProphetwFlags(seas-Mult)":            
            forecast = FORECAST_1036.prophet(salesdata,flags=flagsdata,f=frequency,
                                       seasonality='multiplicative',n=n)
    return forecast 