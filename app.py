import streamlit as st
import joblib
import numpy as np

# Load the model
model = joblib.load("bias_decision_tree_model.pkl")

st.title("🧠 Bias Prediction using Decision Tree")
st.write("Enter numeric features to predict political bias:")

# Input fields
total_votes = st.number_input("Total Votes (standardized)", value=0.0)
agree = st.number_input("Agree (standardized)", value=0.0)
disagree = st.number_input("Disagree (standardized)", value=0.0)
agree_ratio = st.number_input("Agree Ratio (standardized)", value=0.0)
agreeance_score = st.slider("Agreeance Score (0 to 10)", 0, 10, 5)

# Predict
if st.button("Predict Bias"):
    input_data = np.array([[total_votes, agree, disagree, agree_ratio, agreeance_score]])
    prediction = model.predict(input_data)[0]
    st.success(f"🎯 Predicted Bias: **{prediction}**")
