
import pandas as pd
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error

##Expected sales format- pandas series with dates in index 
#Expected format for flags - pd.DataFrame with all dates(Do not split into train and test)
def prophet(sales,n,flags=None,seasonality='additive',f="D",only_error=False,testdata=None):
    """
    sales(pd.Series): sales data with dates in index 
    n(int): No.of periods into future to forecast 
    flags(pd.DataFrame): flags data for data corresponding to sales data AND future forecast periods 
    Seasonality(string("additive"/"multiplicative")): "additive" - to be used for contsant variance in seaonality ; "multiplicative" - to be used for increasing variance in seaonality
    f(string("D","W","M")): frequency of data - daily/weekly/monthly
    only_error(True/ False): Returns only MSE if True ; Returns future forecasts if False
    """
    try:
        #Converting Sales and flags data to format required by Prophet
        if type(sales)==pd.Series:
            #converting pd.Series to pd.Dataframe
            prophdf = pd.Series.to_frame(sales)
        else:
            prophdf = sales
        prophdf["ds"] = prophdf.index
        prophdf.columns = ["y","ds"]
        
        #checking if a df has been passed to flags argument
        if isinstance(flags,pd.DataFrame):
            #Converting flags data to format required by Prophet
            holiday=pd.DataFrame({'holiday':[],'ds': []})
            for i in flags.columns.values[1:]:
                dates=flags[(flags[i]==1).values].index.values
                holidays = pd.DataFrame({'holiday':i,'ds': dates})
                holiday=pd.concat((holiday,holidays))
        
        else:
            holiday=None

        #Instantiating and fitting Prophet
        prop = Prophet(seasonality_mode=seasonality,holidays=holiday,
                       daily_seasonality=False,yearly_seasonality='auto')
        prop.fit(prophdf)     

        #Forecasting 
        future = prop.make_future_dataframe(periods=n,freq=f)
        forecast = prop.predict(future)

        yhat =  forecast['yhat']
        yhat.index = forecast['ds']
        pred = yhat[len(sales):]
        pred.loc[pred<0]=0
        
        if only_error==False:
            return pred
        
        #returns only MSE if only_error=True
        else: 
            error = mean_squared_error(pred,testdata)
            return error

    except:
        print("prophet__An error occured while fitting prophet")
        raise


###Sample code for using prophet from train-test split stage onwards till calculating MSE
#train, test = traintestsplit(data)
#forecast = prophet(train,n=368)
#forecast['yhat']=round(forecast['yhat'])
#mean_squared_error(test,forecast['yhat'])

#forecast.index =forecast['ds']
#plt.plot(forecast['yhat'])
#plt.plot(train)
#plt.show()
#from sklearn.metrics import mean_squared_error
#mean_squared_error(test,prophetforecast['yhat'])

