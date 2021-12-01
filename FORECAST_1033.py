
import pandas as pd
import numpy as np
import keras
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings("ignore")
from calendar import monthrange
import math

"""
#Min Max Scaling Transformation
#col will be a dataframe column containing the values to transformed/inverse-transformed
#origCol will be dataframe with 1 column containing the original min and max before the transformation
#inverse is the paramter to indicate that inverse transformation is required
#Note that origCol and inverse both have to be specified for inverse transformation
#The function will return a transformed or inverse transformed data
"""
def normalize(col , origCol = None , inverse = False):
    try:
        scaler = MinMaxScaler()
        values = col.values
        values = values.reshape((len(values), 1))
        if not inverse:
            scaler.fit(values)
            return scaler.transform(values)
        else:
            oriValues = origCol.values
            oriValues = oriValues.reshape((len(oriValues), 1))
            scaler.fit(oriValues)
            return scaler.inverse_transform(values)
    except:
        print("normalize__An error occurred while applying normalize transformation.")

"""
Splitting the multivariate sequence into samples
"""
def split_sequences(sequences, n_steps_in, n_steps_out):
	X, y = list(), list()
	for i in range(len(sequences)):
		# find the end of this pattern
		end_ix = i + n_steps_in
		out_end_ix = end_ix + n_steps_out
		# check if we are beyond the dataset
		if out_end_ix > len(sequences):
			break
		# gather input and output parts of the pattern
		seq_x, seq_y = sequences[i:end_ix, :], sequences[end_ix:out_end_ix, :]
		X.append(seq_x)
		y.append(seq_y)
	return array(X), array(y)

"""
Training of LSTM model
"""
def LSTMForecast(X , y , nStepsInp , nStepsOut, startMonth):
    try:
        
        X_train, X_test = X[:660,:,:], X[660:,:,:]
        y_train, y_test = y[:660,:,:], y[660:,:,:]
        n_features = X.shape[2]
        #Model Initialization
        model = Sequential()
        model.add(LSTM(100,activation = "relu" , input_shape = (nStepsInp, n_features)))
        model.add(RepeatVector(nStepsOut))
        model.add(LSTM(100, activation = "relu" , return_sequences = True))
        model.add(TimeDistributed(Dense(n_features)))
        model.compile(optimizer = "adam" , loss = "mse")
        
        #Model fitting
        model.fit(X_train , y_train , epochs = 100 , batch_size = 5, verbose = 0)
        
        #Model Predictions
        y_pred = model.predict(X_test, verbose=0)
        
        #Calculating rmse for each set
        rmseLst = []
        for i in range(660,X.shape[0]):
            rmse = mean_squared_error(y_test[i-660],y_pred[i-660]) 
            rmseLst.append(math.sqrt(rmse))
        
        print(rmseLst)
        
        """
        mapeLst = []
        lstDays = [monthrange(2019,m)[1] for m in range(1,13)]
        lst1 = [lstDays[i] for i in range(startMonth,12)]
        lst2 = [lstDays[i] for i in range(startMonth)]
        lstDays = lst1+lst2
        
        
        for i in lstDays:
            X_train, X_test = X[:-(nStepsInp-i),:,:], X[-(nStepsInp-i),:,:]
            y_train, y_test = y[:-(nStepsInp-i),:,:], y[-(nStepsInp-i),:,:]
            n_features = X.shape[2]
            if i==0:
                #Model Initialization
                model = Sequential()
                model.add(LSTM(100,activation = "relu" , input_shape = (nStepsInp, n_features)))
                model.add(RepeatVector(nStepsOut))
                model.add(LSTM(100, activation = "relu" , return_sequences = True))
                model.add(TimeDistributed(Dense(n_features)))
                model.compile(optimizer = "adam" , loss = "mse")
                
                #Model fitting
                model.fit(X_train , y_train , epochs = 100 , batch_size = 5, verbose = 0)
                print("Model trained for iteration : " , (i+1))
                
            else:
                #Model fitting
                model.fit(X_train , y_train , epochs = 100 , verbose = 0)
                print("Model trained for iteration : " , (i+1))
            
            #Model Predictions
            X_test = X_test.reshape((1, nStepsInp, n_features))
            y_pred = model.predict(X_test, verbose=0)
            dfMape = pd.DataFrame(abs((y_pred/y_test-1)).reshape(nStepsOut, n_features))
            mapeLst.append(dfMape.mean())
        
        print(mapeLst)
        """
            
    except:
        print("LSTM__An error occurred while training the LSTM model.")

df = pd.read_excel("C:\\Users\\e1644545\\Desktop\\SHRIJAN\\Data files\\Top_5_SKU.xls")

df1 = df[df["SKU"] == "TR1139C1A1"]
df1.index = pd.to_datetime(df1["Date"])
df1.drop(["SKU","Date"] , inplace = True, axis = 1)

df1["Date"] = df1.index
r = pd.date_range(start=df1["Date"].min(), end=df1["Date"].max())
df1 = df1.set_index('Date').reindex(r).fillna(0).rename_axis('Date').reset_index()
df1.index = df1["Date"]
df1.drop(["Date"] , inplace = True, axis = 1)

#Finding the minimum and maximum of the values for before scaling and creating a dataframe out of it
dfMinMax = pd.DataFrame({"M" : [min(df1["Qty"]) , max(df1["Qty"]) ] })

#Applying min max scaling to the values
scaledValues = normalize(df1["Qty"])

nStepsInp  = 365
nStepsOut  = 90

#Converting the dataset into input/output data format as part of LSTM data preparation
X, y = split_sequences(scaledValues, nStepsInp, nStepsOut)

#LSTMForecast(X , y , nStepsInp , nStepsOut, None)
