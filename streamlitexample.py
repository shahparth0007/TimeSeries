import streamlit as st
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

data1 = pd.read_excel("/Users/parthshah/Downloads/Trial_Data.xlsx")
st.title('Time Series')
# st.dataframe(data=data1, width=None, height=None)

st.write("""
# Explore different Time Series on different datasets
Which one is the best?
""")
def par():
    st.dataframe(data=data1, width=None, height=None)
    
dataset_name = st.sidebar.button('Hit me to Run Package')
print(dataset_name)
if dataset_name:
    par()
