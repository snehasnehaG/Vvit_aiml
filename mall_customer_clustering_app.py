pip install streamlit pandas matplotlib scikit-learn
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Title
st.title("🛍️ Mall Customer Clustering App")

# Load Dataset
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/plotly/datasets/master/mall_customers.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

st.subheader("Dataset Preview")
st.write(df.head())

# Select Features
st.subheader("Select Features for Clustering")

features = st.multiselect(
    "Choose features",
    ["Annual Income (k$)", "Spending Score (1-100)"],
    default=["Annual Income (k$)", "Spending Score (1-100)"]
)

if len(features) != 2:
    st.warning("Please select exactly 2 features for 2D visualization.")
else:
    X = df[features]

    # Choose number of clusters
    k = st.slider("Select number of clusters (K)", 2, 10, 5)

    # Apply KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    y_kmeans = kmeans.fit_predict(X)

    # Plot clusters
    fig, ax = plt.subplots()

    scatter = ax.scatter(
        X.iloc[:, 0],
        X.iloc[:, 1],
        c=y_kmeans
    )

    # Plot centroids
    centroids = kmeans.cluster_centers_
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        s=200,
        marker='X'
    )

    ax.set_xlabel(features[0])
    ax.set_ylabel(features[1])
    ax.set_title("Customer Segments")

    st.pyplot(fig)

    # Insights
    st.subheader("Insights")
    st.write("• Customers are grouped based on income and spending behavior.")
    st.write("• High income + high spending → premium customers")
    st.write("• High income + low spending → target marketing group")
