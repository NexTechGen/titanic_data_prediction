import plotly.express as px
import pandas as pd

data = pd.read_csv("train.csv")

import joblib

import streamlit as st
from PIL import Image

html_temp = """
            <div style="background-color:#6F8FAF;padding:10px">
            <h2 style="color:white;text-align:center; font-size:46px;">Titanic Dataset</h2>
            </div>  """

st.markdown(html_temp, unsafe_allow_html=True)

st.markdown("###")

image = Image.open('titanic.png')
st.image(image)


model = joblib.load('titanic')


tab1, tab2 = st.tabs(["Predictions", "Visualization"])

with tab1:
    col1, col2 = st.columns(2)

    with col1:
        PClass = st.selectbox('Passenger class', ('1st class', '2nd class', '3rd class'))
    with col2:
        gen = st.radio("Sex", ["Male", "Female"])

    age = st.slider('How old are you?', 1, 100, 18)

    col3, col4 = st.columns(2)

    with col3:
        Sib = st.number_input('Number of Siblings/Spouses Aboard', min_value=0, max_value=20, value=2, step=1)
    with col4:
        Parch = st.number_input('Number of Parents/Children Aboard', min_value=0, max_value=20, value=2, step=1)

    fare = st.number_input('Passenger Fare (British pound)')

    embarked = st.radio("Port of Embarkation", ["Southampton", "Cherbourg", "Queenstown"], horizontal=True)

    if PClass == '1st class':
        PClass = 1
    elif PClass == '2nd class':
        PClass = 2
    else:
        PClass = 3

    if gen == 'Male':
        sex = 0
    else:
        sex = 1

    if embarked == 'Southampton':
        emb = 0
    elif embarked == 'Cherbourg':
        emb = 1
    else:
        emb = 2

    # st.write(PClass, sex, age, Sib, Parch, fare, emb)

    input = [[PClass, sex, age, Sib, Parch, fare, emb]]

    pred = model.predict(input)

    if pred == 1:
        status = 'Survived'
    else:
        status = 'Did not survive'

    if st.button("Predict"):
        st.success(f"79.9102132%   &nbsp;&nbsp;&nbsp;  {status}")

with tab2:
    sizes = data.Survived.value_counts()
    labels = ["Survived", "Not survived"]

    # Create a pie chart using Plotly Express
    fig = px.pie(names=labels, values=sizes, title="Survived VS Not Survived")

    # Display the Plotly Express figure in Streamlit
    st.plotly_chart(fig)

    st.write(data)
