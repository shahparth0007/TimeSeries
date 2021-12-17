import streamlit as st
import Starting
import pandas as pd
import os
import pandas as pd 
import numpy as np 


data = pd.read_excel('Sales_Data.xlsx')

st.markdown("<h1 style='text-align: center; color: white;'>Explore different Time Series Algorithms⏳ on a dataset💽 </h1>", unsafe_allow_html=True)   
st.markdown("<h2 style='text-align: center; color: white;'>Which one is the best?🤔  </h2>", unsafe_allow_html=True)   

# st.dataframe(data=data1, width=None, height=None)
st.sidebar.write("Some Buttons to play with")
eda = st.sidebar.button("Tell me About the Data🙉")
timebutton = st.sidebar.button("Run Time Series🏁")
info = st.sidebar.button("Info on Project")

st.sidebar.subheader("""\n\n  Project Members: \n
👩‍💻 Ankana Asit Samantha \n
🧑‍💻 Parth Shah""")

placeholder= st.image("gif-graph-4.gif")


if timebutton:
    placeholder = st.empty()
    algotable,errortable_dict = Starting.forecast(category = 'J', n=3)
    st.write("And The Winner is 🎊 ")
    st.dataframe(data=algotable,  width=None, height=None)
    st.write("Individual Error Tables: ")
    for i in errortable_dict.keys():
        st.write(str(i))
        st.dataframe(data=errortable_dict[str(i)],  width=None, height=None)

if eda:
    st.markdown("<h5>The data is of a Retail Company who wants to predict the sales of data is future so that they can manage their merchandising order</h5>",unsafe_allow_html=True)   
    st.write("Data has: ", data.shape[0], " rows")
    st.write("First date is", data["DATE"].min())
    df = data.rename(columns={'DATE':'index'}).set_index('index')
    st.line_chart(df["SALES"])

if info:
    st.markdown("<h5>Now a days ML in running towards becoming Auto-ML and in fast changing world companies want quick results from multiple methodologies.</h5>",unsafe_allow_html=True)
    st.markdown("<h5>And one such important domain in ML is Timeseries .</h5>",unsafe_allow_html=True)
    st.markdown("<h5>Our future goal is to build a website completely running all available time series model out there and giving out the results on any given dataset.</h5>",unsafe_allow_html=True)   
    st.markdown("What all can our package do right now?")
    st.markdown("* It runs 5 Different Algorithms on a data Provided")
    st.markdown("<h6 style = 'color:green;'> ARIMA: AutoRegressive Integrated Moving Average</h6>",unsafe_allow_html=True)
    st.markdown("<h6 style = 'color:yellow;'> ARIMAX: AutoRegressive Integrated Moving Average with Exogenous variable </h6>",unsafe_allow_html=True)
    st.markdown("<h6 style = 'color:green;'> SARIMA: Seasonal autoregressive integrated moving average</h6>",unsafe_allow_html=True)
    st.markdown("<h6 style = 'color:yellow;'> TSLM: Time Series Linear Model  </h6>",unsafe_allow_html=True)
    st.markdown("<h6 style = 'color:green;'> Prophet: A facebook algorithm </h6>",unsafe_allow_html=True)
    st.markdown("<h6 style = 'color:yellow;'> Holt-Winters </h6>",unsafe_allow_html=True)
    st.markdown("* It takes 2 data files as inputs 1) Sales Data 2) Exogenous variables")
    st.markdown("* We can tune in at any level we want our predction to be weekly, monthly!")
    st.markdown("* Package gives precition on basis of level provided by the user! It can be on store level or store item level or region etc")
    st.markdown("* After Running all algorithms it checks the MAPE value and stores the lowest MAPE value algorithm for future predictions")

    st.markdown("<h6 style = 'color:cyan;'> How to run the Project </h6>",unsafe_allow_html=True)
    st.markdown("* Probably if you are reading this right now you know how to run this file! But we will explain it again!")
    st.markdown("* ")
