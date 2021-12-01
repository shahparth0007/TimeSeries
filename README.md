# TimeSeries
![image](https://user-images.githubusercontent.com/61245297/110231216-4fe36500-7f3c-11eb-8b4f-7664f693f4f4.png)  
A complete Time Series Python Package which helps you in building Models ranging from ARIMA to Prophet 

A time series is a sequence of observations taken sequentially in time.

The primary objective of time series analysis is to develop mathematical models that provide plausible descriptions from sample data.

Note: I have not explained what all algorithms do, if you wish do get a knowledge on this algorithm, do read the links to get an brief idea about the concepts of this algorithms

Details of the Package: 
1) Package include following Algorithms:  
a) ARIMA (Autoregrassive integrated moving average)  
	  https://machinelearningmastery.com/gentle-introduction-box-jenkins-method-time-series-forecasting/  
b) ARIMAX (A more deatil version of ARIMA which also includes independent predictors)  
c) SARIMA (Seasonal Autoregrassive integrated moving average)  
    https://machinelearningmastery.com/sarima-for-time-series-forecasting-in-python/
d) SARIMAX (A more deatil version of SARIMA which also includes independent predictors)  
e) HOLT-Winters  
		https://orangematter.solarwinds.com/2019/12/15/holt-winters-forecasting-simplified/  
f) Time Series Linear Model  
		https://henningsway.rbind.io/post/regression-for-timeseries/  
g) Prophet (Additive Seasonality)  
		https://machinelearningmastery.com/time-series-forecasting-with-prophet-in-python/  
h) Prophet (Multiplicative Seasonaltiy)  		

The Package Excepts 2 Files Sales data and Independent Predictors data  
Sales data Outline Example :  
![image](https://user-images.githubusercontent.com/61245297/110230250-ad27e800-7f35-11eb-8e3a-fbe0002e9338.png)  
Level is the Variable on which we want to do Time Series Forecating it can have multiple values.  
Sales is the Variable which captures the sales(this can be any integer value) of that level on that day.  

Independent Predictors data Outline Example:  
![image](https://user-images.githubusercontent.com/61245297/110230566-e6615780-7f37-11eb-8761-5f1acfbac33e.png)  
Here WEEKEND_FLAG indicates wether that day was a weekend or not. Same with rest flags  

FORECAST_1000.py is the base file which should be executed to run the package.  

A more detail version is coming soon.....


