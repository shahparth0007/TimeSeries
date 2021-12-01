
#Importing the relevant modules
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import acf
from sklearn.metrics import mean_squared_error


#Seasonality const for the various products.Period is in months.
DICT_SEASONALITY_CONST = { "J" : 6 , "E" : 1 , "W" : 12 }

#Augmented Dickey Fuller test for checking the the stationarity in the data
def isStationary(df , significance = 0.05):
    #Store the values present in the desired column into a variable called series
    series = df["Qty"].values
    dftest = adfuller(series, autolag='AIC')
    #Calculating the statistics from the output of dickey fuller test
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    #if p-value is less than significance level then reject the null hypothesis
    #Null hypo is that data is not stationary
    #Alternate hypo is data is stationary
    print(dfoutput[1])
    if dfoutput[1] < significance:
        return True
    else:
        return False


#Returns the seasonality according to the period of the data
def seasonalityPeriod(category, data_freq = "D"):
    m = DICT_SEASONALITY_CONST[category]
    if data_freq == "D":
        m = m * 30
    return m


