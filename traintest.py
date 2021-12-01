
#pass cleaned dataframe as first argument and train-test split propotion as 2nd
#Splits data into train and test set  
def traintestsplit(data,train=0.7):
    try:
        train_size=int(len(data)*train)
        train, test = data.iloc[:train_size,], data.iloc[train_size:,]
        return train,test 
    
    except:
        print("traintestsplit__An error occured while splitting data into train and test")



