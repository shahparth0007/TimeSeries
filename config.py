# -*- coding: utf-8 -*-
import pandas as pd

#Nomenclature for column names
#For Sales data :
#Date column : "DATE" ; ID column : "ID" ; Sales figures column : "SALES"
#For Flags data
#Date column : "DATE"

#Threshold for MSE. If the min MSE of other algos is greater than this value then go for LSTM else do not go for LSTM
MSE_THRESHOLD = 40

#Declaring the constant variables for use throughout the project
DICT_FLAG_PATH = {"J" : ""}

DICT_JEW_PATH = {"J" : "/Users/parthshah/Documents/GitHub/TimeSeries/TimeSeries/Data/train.csv"}

#Dictionary to store the product wise min date to be considered for subsetting data for preprocessing
DICT_START_DATE= { "J" : pd.to_datetime("01/04/2014")}

#Seasonality const for the various products.Period is in months.
DICT_SEASONALITY_CONST = { "J" : 12}

DICT_SEASONALITY = { "J" : {"D" : 365 , "W" : 52 , "M" : 12, "Y" : 1 }}

#TRANSFORMATION TO BE USED ON DATA:
DICT_TRANSFORM = { "L" : "LOG",
                   "D" : "DIFF",
                   "DD": "DOUBLE DIFF", #Difference twice at lag 1
                   "LD": "LOG THEN DIFFERENCE",
                   "E" : "EXPONENTIAL", #e^(-x),
                   "N" : "NORMALIZE",
                   "S" : "STANDARDIZE",
                   "BC": "BOX COX"
                  }



