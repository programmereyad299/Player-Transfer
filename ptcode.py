import numpy as np
import tensorflow as tf
import streamlit as st
import joblib

st.set_page_config(page_title = "Player Transfer Prediction" , page_icon="⚽")
st.title("Player Transfer Prediction ⛹🏻‍♂️⚽")
st.title("")

model = tf.keras.models.load_model('ptmodel.h5')
le1 = joblib.load('ptle1.h5')
le2 = joblib.load('ptle2.h5')
le3 = joblib.load('ptle3.h5')
le12 = joblib.load('ptle12.h5')
ley = joblib.load("ptley.h5")


age = st.slider("Age" , 17 , 39)

nationality  = st.selectbox("Nationality",["Germany","England","France","Portugal","Brazil","Argentina","Netherlands","Spain"])

club  = st.selectbox("Club",["Liverpool","FC Barcelona","Juventus","Manchester City","Bayern Munich","PSG","Real Madrid"])

position  = st.selectbox("Position",["ST","RB","LW","CDM","CM","GK","LB","RW","CB"])

rating = st.slider("Overall Rating" , 60 , 94)

prating = st.slider("Potintial Rating" , 65 , 98)

matches = st.slider("Matches Played" , 0 , 54)

goals = st.slider("Goals" , 0 , 39)

assists = st.slider("Assists" , 0 , 24)

minuites = st.slider("Minuites Played" , 0 , 4497)

value = st.slider("Value in Million Euros" , 0.67 , 180.0)

years = st.slider("Contract Years Left" , 0 , 5)

injury  = st.radio("Injury Prone",["Yes","No"])


nationality = le1.transform([nationality])[0]
club = le2.transform([club])[0]
position = le3.transform([position])[0]
injury = le12.transform([injury])[0]


btn = st.sidebar.button("pred")

if btn :
    input_data = np.array([[age,nationality,club,position,rating,prating,matches,goals,assists
                            ,minuites,value,years,injury]])
    pred = model.predict(input_data)
    pred = np.argmax(pred)  
    pred = ley.inverse_transform([pred])[0]
    st.sidebar.info(pred)