import streamlit as st
import pandas as pd
import numpy as np
import pickle

import os

# Use raw paths or join to avoid escape-sequence issues on Windows
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = pickle.load(open(os.path.join(BASE_DIR, "model.pkl"), "rb"))
model_columns = pickle.load(open(os.path.join(BASE_DIR, "model_columns.pkl"), "rb"))

st.title("🏠 house price predictor")


area =st.number_input("Enter the area of the house in square feet")
bedrooms = st.number_input("Enter the number of bedrooms")
bathrooms = st.number_input("Enter the number of bathrooms")
location = st.slider("Location Rating (1-5)", 1, 5)
age = st.number_input("Enter the age of the house in years")

if st.button("Predict"):
    # Use the same column names as used during training (lowercase)
    input_dict = {
        "area": area,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "age" : age,
        f"Location_{location}": 1
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_dict])

    # Apply same encoding structure
    input_df = pd.get_dummies(input_df) 

    # Align columns
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    prediction = model.predict(input_df)

    st.success(f"Predicted Price: Rs{prediction[0]:,.2f}")