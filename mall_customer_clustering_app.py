 
import streamlit as s
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

st.title("Mall Customer Clustering")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    st.write("Dataset Preview")
    st.write(data.head())

    # Check columns safely
    if data.shape[1] < 5:
        st.error("Dataset must have at least 5 columns")
    else:
        X = data.iloc[:, [3, 4]].values

        # Elbow Method
        wcss = []
        for i in range(1, 11):
            model = KMeans(n_clusters=i, random_state=42)
            model.fit(X)
            wcss.append(model.inertia_)

        fig, ax = plt.subplots()
        ax.plot(range(1, 11), wcss)
        ax.set_title("Elbow Method")
        ax.set_xlabel("Clusters")
        ax.set_ylabel("WCSS")
        st.pyplot(fig)

        k = st.slider("Select clusters", 2, 10, 5)

        model = KMeans(n_clusters=k, random_state=42)
        y = model.fit_predict(X)

        fig2, ax2 = plt.subplots()

        for i in range(k):
            ax2.scatter(X[y == i, 0], X[y == i, 1], label=f"Cluster {i+1}")

        ax2.scatter(model.cluster_centers_[:, 0],
                    model.cluster_centers_[:, 1],
                    marker='X',
                    s=200,
                    label="Centroids")

        ax2.set_xlabel("Income")
        ax2.set_ylabel("Spending Score")
        ax2.legend()

        st.pyplot(fig2)

else:
    st.info("Upload dataset to continue")   
    
