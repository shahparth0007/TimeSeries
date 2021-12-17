#Importing the relevant modules
import pandas as pd
import config
import numpy as np  
import itertools

#*********Function List*********#
#1)selectFileFromPath : Loads file 
#2)sanitizeDates : Uniform dates
#3)dataExtraction : 

#This function will be used to select a csv/excel file from the path which is passed to it
def selectFileFromPath(strPath):
    try:
        if strPath.endswith("csv"):
            df = pd.read_csv(strPath, parse_dates = True, dayfirst=True)
        elif strPath.endswith("xlsx"):
            df = pd.read_excel(strPath)#, parse_dates = True, dayfirst=True)
        elif strPath.endswith("xls"):
            df = pd.read_excel(strPath)#, parse_dates = True, dayfirst=True)            
            
        return df
    
    except:
        print("selectFileFromPath__An error occurred while fetching file from path.")


#This function will sanitize the dataframe by making all the DATE formats uniform in the dataframe passed to it
def sanitizeDates(df):
    try:
        df['DATE'] = df['DATE'].apply(lambda x: pd.to_datetime(x, format='%d-%m-%Y'))
        return df
    except:
        print("sanitizeDates__An error occurred while sanitizing the dataframe.")


#This function will take multiple inputs from user and return a merged dataframe of sales data and flag data
#level can be :- { OVR : Overall ; REG : Regional ; STO : Store ; SKU : SKU ; CAT : Category ; CAP : Category + Price Band }
def dataExtraction(category , level , salesFromPath = True , flagsFromPath = True):
    
    try:

        if salesFromPath == True and flagsFromPath == True:
            print("Started")
            salesDF , flagsDF = extractdata(config.DICT_JEW_PATH[category] , config.DICT_FLAG_PATH[category] , level)
            dictColNames = {"OVR" : "Overall" , "REG" : "Regional" , "STO" : "Store" , "SKU" : "SKU" , "CAT" : "Category" , "CAP" : "CAT PB" }                          
            return salesDF , flagsDF
    except:
        print("dataExtraction__An error occurred while extracting and merging the 2 dataframes.")
        raise


def expandgrid(*itrs):
    product = list(itertools.product(*itrs))
    return {'Var{}'.format(i+1):[x[i] for x in product] for i in range(len(itrs))}

#level parameter must be case sensitive to the column name in the csv/excel file 
def extractdata(salesdata_filepath,flags_filepath,level='ID' , salesDF = None,freq="D"):
    print("Extraction_started")
    level=str(level)
    #importing flags and sales data 
    #flags=pd.read_csv(str(flags_filepath))
    flags = selectFileFromPath(str(flags_filepath))
    print("Flags data shape:", flags.shape)
    
    if not salesdata_filepath == "":
        #salesdata = pd.read_csv(str(salesdata_filepath))
        salesdata = selectFileFromPath(str(salesdata_filepath))
    else:
        salesdata = salesDF
    
    print("Sales data shape:", salesdata.shape)
    #Converting DATE in string format to datetime format 
    flags['DATE'] = pd.to_datetime(flags['DATE'],format='%d-%m-%Y')
    salesdata['DATE'] = pd.to_datetime(salesdata['DATE'],format='%d-%m-%Y')

    #Extracting min and max DATE from dataset and creating index with all dates in between 
    mindate=salesdata['DATE'].min()
    print("Min_data",mindate)
    maxdate=salesdata['DATE'].max()
    print("Max_data",maxdate)
    date_rng = pd.date_range(start=mindate, end=maxdate, freq=freq)

    #Extracting all unique regions/stores/cat/PB
    unique=set(salesdata.loc[:,level])

    #Creating df with all combinations of level and DATE
    completedata=expandgrid(unique,date_rng)
    completedf=pd.DataFrame(completedata)

    #Appropriately renaming columnns of new df
    completedf=completedf.rename(columns={'Var1':'ID','Var2':'DATE'})
    
    #Merging new df with flags and sales dataset 
    #completedf=pd.merge(flags,completedf,on='DATE')
    completedf=pd.merge(completedf,salesdata,how='left',left_on=["ID",'DATE'],right_on=[level,'DATE'])
    
    #Setting DATE as index of df 
    completedf = completedf.set_index('DATE')
    #completedf=completedf.drop(level,axis=1)
    #print(completedf.head())
    completedf.columns=['ID',"SALES"]
    flags = flags.set_index('DATE')
    
    #Replacing na with 0 
    sales=completedf.fillna(0)
    flags=flags.fillna(0)
    return sales, flags

#extractdata("Top_5_SKU.xls","All Flags.csv","SKU")

