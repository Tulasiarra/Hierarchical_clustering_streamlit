import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import plotly.express as px

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="Hierarchical Clustering Dashboard",
    page_icon="🌳",
    layout="wide"
)

# ================= LOAD DATA =================
@st.cache_data
def load_data():
    # Raw dataset (same as Jupyter)
    df = pd.read_csv("Mall_Customers.csv")

    # Clustered dataset (generated in Jupyter)
    clustered_df = pd.read_csv("Hierarchical_clustering.csv")

    return df, clustered_df

df, clustered_df = load_data()

# ================= FEATURE COLUMNS =================
raw_features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
cluster_features = ["Age", "Annual Income", "Spending Score"]

# ================= HEADER =================
st.markdown(
    "<h1 style='text-align:center;'>🌳 Hierarchical Clustering Dashboard</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align:center;'>Mall Customer Segmentation using Hierarchical Clustering</p>",
    unsafe_allow_html=True
)
st.markdown("---")

# ================= SIDEBAR =================
st.sidebar.title("⚙️ Controls")

linkage_method = st.sidebar.selectbox(
    "Select Linkage Method",
    ["single", "complete", "average", "ward"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Hierarchical Clustering Types**
    - Agglomerative (Bottom-Up)
    - Divisive (Top-Down)
    """
)

# ================= SECTION 1: TYPES =================
st.subheader("📌 Types of Hierarchical Clustering")

c1, c2 = st.columns(2)

with c1:
    st.success(
        """
        **Agglomerative Clustering**
        - Bottom-up approach  
        - Each data point starts as a cluster  
        - Closest clusters merge iteratively  
        - Most commonly used method  
        """
    )

with c2:
    st.warning(
        """
        **Divisive Clustering**
        - Top-down approach  
        - Starts with one large cluster  
        - Splits into smaller clusters  
        - Computationally expensive  
        """
    )

# ================= SECTION 2: DENDROGRAM =================
st.subheader("🌲 Dendrogram Visualization")

X = df[raw_features]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

Z = linkage(X_scaled, method=linkage_method)

fig, ax = plt.subplots(figsize=(14, 5))
dendrogram(Z, ax=ax)
ax.set_title(f"Dendrogram ({linkage_method.capitalize()} Linkage)")
ax.set_xlabel("Customers")
ax.set_ylabel("Distance")

st.pyplot(fig)

st.info("✂️ Cut the dendrogram at a suitable distance to decide the number of clusters.")

# ================= SECTION 3: LINKAGE METHODS =================
st.subheader("🔗 Linkage Methods")

st.markdown(
    """
    - **Single Linkage** → Minimum distance between clusters  
    - **Complete Linkage** → Maximum distance between clusters  
    - **Average Linkage** → Mean distance between clusters  
    - **Ward’s Method** → Minimizes variance within clusters (best performance)  
    """
)

# ================= SECTION 4: DECISION TREE STYLE FLOW =================
st.subheader("🌳 Hierarchical Clustering Flow")

st.code(
"""
Start
 └── Each customer is an individual cluster
     └── Compute distance between clusters
         └── Apply linkage method
             └── Merge closest clusters
                 └── Repeat until one cluster remains
                     └── Visualize using dendrogram
""",
language="text"
)

# ================= SECTION 5: CLUSTER VISUALIZATION =================
st.subheader("📊 Customer Segmentation")

fig_scatter = px.scatter(
    clustered_df,
    x="Annual Income",
    y="Spending Score",
    color="Cluster",
    title="Hierarchical Clustering – Mall Customers",
    template="plotly_white"
)

st.plotly_chart(fig_scatter, use_container_width=True)

# ================= SECTION 6: PREDICTION =================
st.subheader("🎯 Predict Cluster for New Customer")

c1, c2, c3 = st.columns(3)

with c1:
    age = st.slider("Age", 18, 70, 30)

with c2:
    income = st.slider("Annual Income", 15, 140, 50)

with c3:
    score = st.slider("Spending Score", 1, 100, 50)

if st.button("🚀 Predict Cluster"):

    input_df = pd.DataFrame(
        [[age, income, score]],
        columns=cluster_features
    )

    scaler_pred = StandardScaler()
    X_clustered = scaler_pred.fit_transform(clustered_df[cluster_features])
    X_input = scaler_pred.transform(input_df)

    centroids = (
        pd.DataFrame(X_clustered, columns=cluster_features)
        .groupby(clustered_df["Cluster"])
        .mean()
    )

    distances = pairwise_distances(X_input, centroids)
    cluster = centroids.index[distances.argmin()]

    st.success(f"🎯 Predicted Cluster: **{cluster}**")

    cluster_data = clustered_df[clustered_df["Cluster"] == cluster]

    m1, m2, m3 = st.columns(3)
    m1.metric("Customers in Cluster", len(cluster_data))
    m2.metric("Avg Income", f"{cluster_data['Annual Income'].mean():.2f}")
    m3.metric("Avg Spending", f"{cluster_data['Spending Score'].mean():.2f}")

# ================= FOOTER =================
st.markdown("---")
st.markdown(
    "<p style='text-align:center;'>Hierarchical Clustering • Streamlit • Machine Learning</p>",
    unsafe_allow_html=True
)
