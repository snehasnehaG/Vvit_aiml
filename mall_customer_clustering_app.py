import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Title
st.title("🛍️ Mall Customer Clustering App")

# Load dataset
url = "https://raw.githubusercontent.com/plotly/datasets/master/mall_customers.csv"
df = pd.read_csv(url)

# Show data
st.subheader("Dataset Preview")
st.write(df.head())

# Select features
st.subheader("Select Features for Clustering")
feature1 = st.selectbox("Select X-axis feature", df.columns[1:])
feature2 = st.selectbox("Select Y-axis feature", df.columns[1:])

X = df[[feature1, feature2]]

# Choose number of clusters
k = st.slider("Select number of clusters (K)", 2, 10, 3)

# Apply KMeans
model = KMeans(n_clusters=k, random_state=42)
df["Cluster"] = model.fit_predict(X)

# Show clustered data
st.subheader("Clustered Data")
st.write(df.head())

# Plot clustering
st.subheader("Customer Segments (Scatter Plot)")
fig, ax = plt.subplots()
scatter = ax.scatter(X[feature1], X[feature2], c=df["Cluster"])
ax.set_xlabel(feature1)
ax.set_ylabel(feature2)
ax.set_title("Customer Clusters")
st.pyplot(fig)

# Histogram
st.subheader("Histogram")
fig2, ax2 = plt.subplots()
ax2.hist(X[feature1])
ax2.set_title(f"Distribution of {feature1}")
st.pyplot(fig2)

