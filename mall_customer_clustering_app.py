import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Title
st.title("Mall Customer Clustering App")

# Upload dataset
uploaded_file = st.file_uploader("Upload Mall_Customers.csv", type=["csv"])

if uploaded_file is not None:
    dataset = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.write(dataset.head())

    # Select features
    X = dataset.iloc[:, [3, 4]].values

    # Elbow Method
    st.subheader("Elbow Method")
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
        kmeans.fit(X)
        wcss.append(kmeans.inertia_)

    fig1, ax1 = plt.subplots()
    ax1.plot(range(1, 11), wcss)
    ax1.set_title("Elbow Method")
    ax1.set_xlabel("Number of clusters")
    ax1.set_ylabel("WCSS")
    st.pyplot(fig1)

    # Choose clusters
    k = st.slider("Select number of clusters", 2, 10, 5)

    # KMeans
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    y_kmeans = kmeans.fit_predict(X)

    # Visualization
    st.subheader("Customer Clusters")

    fig2, ax2 = plt.subplots()

    for i in range(k):
        ax2.scatter(X[y_kmeans == i, 0],
                    X[y_kmeans == i, 1],
                    s=50,
                    label=f"Cluster {i+1}")

    # Centroids
    ax2.scatter(kmeans.cluster_centers_[:, 0],
                kmeans.cluster_centers_[:, 1],
                s=200,
                marker='X',
                label='Centroids')

    ax2.set_title("Clusters of Customers")
    ax2.set_xlabel("Annual Income")
    ax2.set_ylabel("Spending Score")
    ax2.legend()

    st.pyplot(fig2)

else:
    st.info("Please upload a CSV file to proceed.")
