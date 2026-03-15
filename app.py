import streamlit as st
import numpy as np
import pickle

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

# Page title
st.title("🎓 College Admission Prediction")
st.write("Enter student details to predict the chance of admission.")

st.divider()

# User Inputs
gre_score = st.number_input("GRE Score", min_value=0, max_value=340, value=300)
toefl_score = st.number_input("TOEFL Score", min_value=0, max_value=120, value=100)
university_rating = st.slider("University Rating", 1, 5, 3)
sop = st.slider("Statement of Purpose Strength", 1.0, 5.0, 3.0)
lor = st.slider("Letter of Recommendation Strength", 1.0, 5.0, 3.0)
cgpa = st.number_input("CGPA", min_value=0.0, max_value=10.0, value=8.0)
research = st.selectbox("Research Experience", [0, 1])

st.divider()

# Predict Button
if st.button("Predict Admission Chance"):

    input_data = np.array([[gre_score, toefl_score, university_rating, sop, lor, cgpa, research]])

    prediction = model.predict(input_data)

    chance = float(prediction[0]) * 100

    st.success(f"🎯 Predicted Chance of Admission: {chance:.2f}%")

    if chance > 75:
        st.write("✅ High chance of admission!")
    elif chance > 50:
        st.write("⚠️ Moderate chance of admission.")
    else:
        st.write("❌ Low chance of admission.")
