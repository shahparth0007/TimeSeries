"""
############################################ CODED BY SHRIJAN ############################################
"""

#Importing the relevant modules
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import config

"""
#Log & Inverse Log Transformation
#Col is the dataframe column to be transformed/inverse-transformed
#inverse is the paramter to indicate that inverse transformation is required
#The function will return a transformed or inverse transformed data
"""
def logTrans(col, inverse = False):
    try:
        if not inverse:
            return np.log(col.values)
        else:
            return np.exp(col.values)
            
    except:
        print("logTrans__An error occurred while applying log transformation.")

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
#This function will subset the dataframe based on the min date criteria for that product (J/E/W)
Takes a data frame and a product category i.e "J"/"E"/"W" as input and subsets the data
"""
def subsetDataframe(df , category):
    try:
        #Subsetting the dataframe to get before the category's start date data
        subsetDF = df[df.index <= config.DICT_START_DATE[category]]
        
        #Getting the list of unique SKUs/Stores/CAT/CATPB for the above df
        idListInSubsetDF = subsetDF["ID"].unique()
        
        #Setting the flag only for those rows whose data was present in the subsetted dataframe
        boolCond = [ df.loc[i,"ID"] in idListInSubsetDF for i in range(df.shape[0])]
        
        return df[boolCond]
        
    except:
        print("subsetDataframe__An error occurred while subsetting the data.")
    return df


"""
Created on Tue Dec  3 10:01:55 2019

@author: Hardik
"""

#import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

from statsmodels.tsa.stattools import adfuller
from scipy.special import inv_boxcox
from tslearn.preprocessing import TimeSeriesScalerMeanVariance as standardize

#Splits data into train and test set  
def traintestsplit(data,train=0.7):
    try:
        train_size=int(len(data)*train)
        train, test = data.iloc[:train_size,], data.iloc[train_size:,]
        return train,test 
    
    except:
        print("traintestsplit__An error occured while splitting data into train and test")


#Transformation for standardising data   
#def standardize(data,inverse=False,inv=0):
#    try:
#        values = data.values
#        values = values.reshape((len(values), 1))
#        scaler = StandardScaler()
#        if inverse==False:
#            scaler = scaler.fit(values)
#            standardized = scaler.transform(values)
#           return standardized
#       else:
#            invrs = inv.values
#            invrs = invrs.reshape((len(values), 1))
#            scaler = scaler.fit(values)
#            inversed = scaler.inverse_transform(invrs)
#            return inversed#
#
#    except:
#        print("standardize__An error occured while standardizing data")
        

#DIFFERENCING
# create a differenced series
def difference(dataset, interval=1,inverse=False,initial_value=0):
    try:
        if inverse==False:
            diff = list()
            for i in range(interval, len(dataset)):
                value = dataset[i] - dataset[i - interval]
                diff.append(value)
            return pd.Series(diff)
        else:
            inv=list()
            inv.append(initial_value)
            for i in range(len(dataset)):
                value = inv[i] + dataset[i]
                inv.append(value)
            return pd.Series(inv)
    
    except:
        print("difference__An error occured while differencing data")
        
        
        
#Applying various transformations to data to stationarize it 
#l-log,d-difference,dd- double difference, ld- log+difference, e- exponential(e^-x) 
# n- normalize, s-standardize, bc-boxcox
def transform(data,conflev=0.95,transformation="d"):  
    try:
        #1st order difference
        if str(transformation)=="d":
            diffdata=difference(data)
            output=adfuller(diffdata)
            pval = output[1]
            print("p-value of Augmented Dickey Fuller Test using this transformation is",pval)
            initial_value= data[-1]
            return diffdata,transformation, initial_value
        
        #2nd order difference
        if str(transformation)=="dd":
            diffdata=difference(data)
            diff2data=difference(diffdata)
            output=adfuller(diff2data)
            pval = output[1]
            print("p-value of Augmented Dickey Fuller Test using this transformation is",pval)
            initial_values= [data[-1],diffdata[-1]]
            return diff2data,transformation, initial_values
        
        #Log transformation 
        if str(transformation)=="l":
            logdata=np.log(data)
            output=adfuller(logdata)
            pval = output[1]   
            print("p-value of Augmented Dickey Fuller Test using this transformation is",pval)
            return logdata,transformation
        
        #log+differencing
        if str(transformation)=="ld":
            logdata=np.log(data)
            difflogdata=difference(logdata)
            output=adfuller(difflogdata)
            pval = output[1]    
            print("p-value of Augmented Dickey Fuller Test using this transformation is",pval)
            intial_value = logdata[-1] 
            return difflogdata, transformation, intial_value
    
        #exponential 
        if str(transformation)=="e":
            data = -data
            expdata= np.exp(data)
            output=adfuller(expdata)
            pval = output[1]    
            print("p-value of Augmented Dickey Fuller Test using this transformation is",pval)
            return expdata, transformation
        
        #Box-Cox
        if str(transformation)=="bc":
            boxcoxdata  = stats.boxcox(data)
            output=adfuller(boxcoxdata[0])
            pval = output[1]    
            print("p-value of Augmented Dickey Fuller Test using this transformation is",pval)
            return boxcoxdata[0], transformation, boxcoxdata[1]
        
        #Normalizing
        if str(transformation)=="n":
            normdata=normalize(data)
            minmaxvalues = [np.min(data),np.max(data)]
            return normdata, transformation, minmaxvalues
        
        #Standardizing 
        if str(transformation)=="s":
            standata = standardize(0,1).fit_transform(data)
            standata = pd.Series(np.transpose(standata[:,:,0])[:,0])
            standata.index = data.index
            mean = np.mean(data)
            sd = np.std(data)
            param = [mean,sd]
            return standata,transformation,param   
    
    except:
        raise
        print("transform_An error occured while transforming data")
    
""
# Returns a tuple with 1st element- transformed data, 2nd element- name of 
# transformation and variable 3rd element- parameters required for inversing transformation 
""


""
# 1st argument should be forecasted scaled values that are to be inversed
# 2nd argument should be entire tuple that was returned from the function- transform 
""
def inverse(forecastvalues, transformeddata):
    try:
        if transformeddata[1]=="d":
            inversed = difference(forecastvalues, inverse=True, initial_value=transformeddata[2])
            return inversed 
        if transformeddata[1]=="dd":
            inversed1 = difference(forecastvalues, inverse=True, initial_value=transformeddata[2][1])
            inversed2 = difference(inversed1, inverse=True, initial_value=transformeddata[2][0])
            return inversed2
        if transformeddata[1]=="l":
            inversed = np.exp(forecastvalues)
            return inversed
        if transformeddata[1]=="ld":
            inversed1 = difference(forecastvalues, inverse=True, initial_value=transformeddata[2])
            inversed2 = np.exp(inversed1)
            return inversed2
        if transformeddata[1]=="n":
            inversed = normalize(forecastvalues, transformeddata[2], inverse=True)
            return inversed 
        if transformeddata[1]=="s":
            inversed = standardize(transformeddata[2][0],transformeddata[2][1]).fit_transform(forecastvalues)
            inversed = pd.Series(np.transpose(inversed[:,:,0])[:,0])
            inversed.index = forecastvalues.index
            return inversed 
        if transformeddata[1]=="bc":
            inversed = inv_boxcox(forecastvalues,transformeddata[2])
            
    except:
        raise
        print("inverse_An error occured while inversing transformation")



#Applying various transformations to data to stationarize it 
def stationarize(data,conflev=0.95):    
    #1st order difference
    diffdata=difference(data)
    output=adfuller(diffdata)
    pval = output[1]
    #2nd order difference
    if pval>=(1-conflev):
        diff2data=difference(diffdata)
        output=adfuller(diff2data)
        pval = output[1]
        if pval<(1-conflev):
            print("Data transformation: 2nd order differencing") 
            transformation = "dd"
            initial_values= [data[0],diffdata[0]]
            return diff2data,transformation, initial_values
    else:
        print("Data transformation: 1st order differencing")
        transformation = "d"
        initial_value= data[0]
        return diffdata,transformation, initial_value

    #Log transformation 
    logdata=logTrans(data)
    output=adfuller(logdata)
    pval = output[1]
    #log+differencing
    if pval>=(1-conflev):
        difflogdata=difference(logdata)
        output=adfuller(difflogdata)
        pval = output[1]
        if pval<(1-conflev):
            print("Data transformation: Log + 1st order differencing") 
            transformation = "ld"
            intial_value = logdata[0] 
            return difflogdata, transformation, intial_value
    else:
        print("Data transformation: Log")
        transformation = "log" 
        return logdata,transformation
    
    #Normalizing 
    normdata=normalize(data)
    output=adfuller(normdata)
    pval = output[1]       
    if pval<(1-conflev):
        print("Data transformation: Normalization") 
        transformation = "n"
        minmaxvalues = [np.min(data),np.max(data)]
        return normdata,transformation, minmaxvalues
    
    #Standardizing 
    standata = standardize(0,1).fit_transform(data)
    standata = pd.Series(np.transpose(standata[:,:,0])[:,0])
    output=adfuller(standata)
    pval = output[1]       
    if pval<(1-conflev):
        print("Data transformation: standardization") 
        transformation = "s"
        standata.index = data.index
        mean = np.mean(data)
        sd = np.std(data)
        param = [mean,sd]
        return standata[0],transformation, param
    
"""
# Returns a tuple with 1st element- transformed data, 2nd element- name of 
# transformation and variable 3rd element- parameters required for inversing transformation 
"""
