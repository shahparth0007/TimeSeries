
from pmdarima.arima import auto_arima
from transform import difference, transform
 
#Full data(in form of pd.Series) should be passed. Not train/test data 
def sarimax(data,length,f,n=1,Flags=None,only_error=False,exog=None):
    """
    data(pd.Series): data in form of pandas series 
    f-frequency(string): "D" = day level data ; "W" = weekly level data ; "M" = Monthly level data
    length(int): No.of data points to be used for testing 
    n(int): no.of periods ahead to forecast 
    Flags(pd.DataFrame): flags data corresponding to dates in 'data '
    exog(pd.DataFrame): flags data corresponding to dates for forecast period 
    only_error(True/False): Returns only MSE and test data predictions if True ; Returns future forecasts if False
    """   
    if f=="W":
        frequency = 52
    elif f=="M":
        frequency = 12
    model = auto_arima(data,exogenous=Flags,m=frequency,information_criterion='oob',
                       out_of_sample_size=length,scoring='mse',seasonal=True)
        
    if only_error==True:
        error = model.oob_
        pred = model.oob_preds_
        return error ,pred

    if only_error==False:
        preds = model.predict(n_periods=n,exogenous=exog)
        return preds 
        
            
        

        
