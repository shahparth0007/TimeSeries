
import pandas as pd
import numpy as np 
from tslearn.clustering import TimeSeriesKMeans
import matplotlib.pyplot as plt 
from pyclustering.cluster.silhouette import silhouette_ksearch_type, silhouette_ksearch

## Testing clusterig with CatPB data
#Importing data and filling in missing values  
#data = pd.read_csv("Catpb_plain_weight_data.csv")
#df = data.iloc[:,1:]
#df = df.fillna(0)
#df = np.array(df)
#df = list(df)

#df must be a pd dataframe with sku/store/catPB and sales columns  
def cluster(df,freq):
    df['DATE']=df.index
    df = df.pivot(index="ID",columns='DATE',values='SALES')
    
    #converts daily to weekly data as a form of dimensionality reduction
    if freq=="D":
        df = pd.DataFrame(np.add.reduceat(df.values, np.arange(len(df.columns))[::7], axis=1))
    
    dfl=list(np.array(df)[:,350:])
    return dfl
    #Searching for optimal value of k(No.of clusters)
    search_instance = silhouette_ksearch(dfl, 2, 10, algorithm=silhouette_ksearch_type.KMEANS).process()
    amount = search_instance.get_amount()
    
    clusters = TimeSeriesKMeans(n_clusters=amount, metric="euclidean",random_state=2).fit_predict(dfl)
    return clusters,df

"""
a=search_instance_instance._silhouette_ksearch__calculate_clusters(2)

catPB = Catpb_plain_weight_datacsv.iloc[:,1:]
catPB = catPB.fillna(0)
"""

dfp = np.transpose(df.iloc[:,350:])


"""
#Plotting clustered time series 
plt.plot(df.iloc[:,list(km==2)],c='blue',alpha=0.5)
plt.plot(df.iloc[:,list(km==1)],c='black',alpha=0.5)
plt.plot(df.iloc[:,list(km==4)],c='red',alpha=0.5)
plt.plot(df.iloc[:,list(km==3)],c='purple',alpha=0.5)
plt.plot(df.iloc[:,list(km==0)],c='pink')
plt.show()
"""