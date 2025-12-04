import streamlit as st
import pandas as pd
import pickle

# Load model
model = pickle.load(open("modelKNN1.pkl", "rb"))

st.title("Iris Flower Prediction")

sepal_length = st.number_input("Sepal Length")
sepal_width = st.number_input("Sepal Width")
petal_length = st.number_input("Petal Length")
petal_width = st.number_input("Petal Width")

if st.button("Predict"):
    # kolom HARUS sama dengan saat model.fit(...)
    input_data = pd.DataFrame(
        [[sepal_length, sepal_width, petal_length, petal_width]],
        columns=["sepal_length", "sepal_width", "petal_length", "petal_width"]
    )

    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Species: {prediction}")
