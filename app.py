import streamlit as st
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import base64
import time

# ------------------------- SETUP -------------------------
st.set_page_config(page_title="HCM KMeans Clustering", layout="wide")

# Load KMeans Model
with open("kmean.pkl", "rb") as file:
    kmeans = pickle.load(file)

# Define Feature Names – MUST match training features
FEATURE_NAMES = [
    'Birth Rate', 'Business Tax Rate', 'CO2 Emissions',
    'Days to Start Business', 'Energy Usage', 'GDP', 'Health Exp % GDP',
    'Health Exp/Capita', 'Hours to do Tax', 'Infant Mortality Rate',
    'Internet Usage', 'Lending Interest', 'Life Expectancy Female',
    'Life Expectancy Male', 'Mobile Phone Usage', 'Population 0-14',
    'Population 15-64', 'Population 65+', 'Population Total',
    'Population Urban', 'Tourism Inbound', 'Tourism Outbound'
]

# Background Image Setup
def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Optional: change to relative path or remove if deploying
bg_image_path = r"C:\Users\saite\OneDrive\Desktop\cluster\59874.jpg"
  # Update this to your image file name in same folder

try:
    base64_img = get_base64_image(bg_image_path)
    st.markdown(f"""
        <style>
            .stApp {{
                background-image: url("data:image/jpg;base64,{base64_img}");
                background-size: cover;
                background-position: center;
                background-repeat: no-repeat;
            }}
        </style>
    """, unsafe_allow_html=True)
except:
    st.warning("⚠️ Background image not found. Please check the path.")

# --------------------- Title ---------------------
st.markdown(
    """
    <div style="background-color: #2C3E50; padding: 15px; border-radius: 12px; text-align: center;">
        <h1 style="color: white; font-size: 32px;">🌍 HCM KMeans Cluster Prediction</h1>
    </div>
    """, unsafe_allow_html=True
)

# ---------------- Sidebar Input ------------------
st.sidebar.header("Input Features")
user_input = [st.sidebar.number_input(feature, min_value=0.0, step=0.5) for feature in FEATURE_NAMES]
input_array = np.array(user_input).reshape(1, -1)

# ------------------ Prediction -------------------
st.markdown("<div style='display: flex; justify-content: center;'>", unsafe_allow_html=True)
if st.button("Predict 🚀"):
    if input_array.shape[1] != kmeans.n_features_in_:
        st.error(f"⚠️ Expected {kmeans.n_features_in_} features but got {input_array.shape[1]}.")
    else:
        cluster = kmeans.predict(input_array)[0]
        progress_bar = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress_bar.progress(i + 1)

        st.markdown(
            f"""
            <div style='background-color: #222831; padding: 15px; border-radius: 12px; text-align: center;'>
                <h2 style='color: white;'>🎯 Predicted Cluster: {cluster}</h2>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.snow()

# ----------------- File Upload ------------------
col1, col2 = st.columns(2)
with col1:
    st.title("Upload Excel File")
    uploaded_file = st.file_uploader("Choose a .xlsx file", type=["xlsx"])
    if uploaded_file:
        try:
            df = pd.read_excel(uploaded_file, engine='openpyxl')
            st.success("✅ File uploaded successfully.")
            st.write(df.head())
        except Exception as e:
            st.error(f"❌ Failed to read Excel file: {e}")

with col2:
    st.title("Data Summary")
    if 'df' in locals():
        st.write(df.describe())

# --------------- Visualization Section -------------
if 'df' in locals():
    col3, col4 = st.columns(2)
    with col3:
        st.title("Histogram")
        column = st.selectbox("Choose column for Histogram", df.columns)
        fig, ax = plt.subplots()
        ax.hist(df[column].dropna(), bins=20, edgecolor='black')
        ax.set_title(f"Histogram of {column}")
        st.pyplot(fig)

    with col4:
        st.title("Box Plot")
        column = st.selectbox("Choose column for Box Plot", df.columns, key="box")
        fig, ax = plt.subplots()
        sns.boxplot(x=df[column], ax=ax)
        ax.set_title(f"Box Plot of {column}")
        st.pyplot(fig)

    # Scatter & Heatmap
    col5, col6 = st.columns(2)
    with col5:
        st.title("Scatter Plot")
        x_col = st.selectbox("X-axis", df.columns, key="scatter_x")
        y_col = st.selectbox("Y-axis", df.columns, key="scatter_y")
        fig, ax = plt.subplots()
        ax.scatter(df[x_col], df[y_col], alpha=0.6)
        ax.set_title(f"{x_col} vs {y_col}")
        st.pyplot(fig)

    with col6:
        st.title("Correlation Heatmap")
        num_df = df.select_dtypes(include=[np.number])
        if num_df.shape[1] >= 2:
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        else:
            st.warning("Not enough numeric columns for correlation heatmap.")
else:
    st.info("📥 Upload an Excel file to enable visualizations.")