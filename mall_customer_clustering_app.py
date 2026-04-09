import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Title
st.title("🛍️ Mall Customer Clustering App")

# Description
st.write("Enter customer details to see their income and spending score.")

# Input: Annual Income
income = st.number_input("Enter Annual Income (k$)", min_value=0, max_value=200)

# Input: Spending Score
spending_score = st.slider("Select Spending Score (1-100)", 1, 100)

# Button
if st.button("Submit"):
    st.success(f"Income: {income} k$ | Spending Score: {spending_score}")

# Checkbox example
if st.checkbox("Show customer message"):
    st.write("Customer data recorded successfully.")

# Simple chart
data = pd.DataFrame(
    np.random.randn(20, 2),
    columns=['Income', 'Spending Score']
)

st.line_chart(data) 

   
        

    
