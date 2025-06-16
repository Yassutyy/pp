import streamlit as st
import pandas as pd
import pickle
import plotly.express as px

# Load encoders and models
with open("brand_encoder.pkl", "rb") as f:
    brand_encoder = pickle.load(f)
with open("fuel_encoder.pkl", "rb") as f:
    fuel_encoder = pickle.load(f)
with open("model_lr.pkl", "rb") as f:
    model_lr = pickle.load(f)
with open("model_rf.pkl", "rb") as f:
    model_rf = pickle.load(f)

# Load dataset
df = pd.read_csv("car_data_set.csv")

# Page configuration
st.set_page_config(layout="wide")

# Sidebar
st.sidebar.title("🧭 Navigation")
option = st.sidebar.radio("Go to", ["🏠 Home", "📁 Dataset", "📊 Visualizations", "🧠 Predictor"])

# Main Title
st.markdown("<h1 style='text-align: center; color: #2C3E50;'>🚗 Car Price Prediction Tool</h1>", unsafe_allow_html=True)

# Home Page
if option == "🏠 Home":
    st.markdown("""
    ### 🔧 About This Tool
    This application helps you:
    - 📁 View the training dataset
    - 📊 Explore data visualizations
    - 🧠 Predict car prices using:
        - Linear Regression
        - Random Forest Regression

    Use the sidebar to navigate ➡️
    """)
    st.caption("Developed by B.Yaswanth, A.Dinesh, SK.Baji")

# Dataset Page
elif option == "📁 Dataset":
    st.subheader("📂 Training Dataset")
    st.dataframe(df)

# Visualization Page
elif option == "📊 Visualizations":
    st.subheader("📊 Data Visualizations")

    fig1 = px.histogram(df, x='Selling_Price', nbins=50, title='Selling Price Distribution', marginal="box")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.box(df, x='Fuel', y='Selling_Price', title='Selling Price by Fuel Type')
    st.plotly_chart(fig2, use_container_width=True)

# Predictor Page
elif option == "🧠 Predictor":
    st.subheader("⚙️ Choose Prediction Model")
    model_choice = st.radio("Select Model", ["Linear Regression - 0.31 (R2 Score) ", "Random Forest - 0.64 (R2 Score)"])

    st.markdown("### 📝 Input Car Details")
    brand = st.selectbox("Select Car Brand", df['Brand'].unique())
    year = st.slider("Manufacturing Year", 1995, 2025, 2015)
    km_driven = st.number_input("Kilometers Driven", min_value=0, max_value=500000, value=30000)
    fuel = st.selectbox("Select Fuel Type", df['Fuel'].unique())

    if st.button("🚀 Predict Selling Price"):
        try:
            brand_encoded = brand_encoder.transform([brand])[0]
            fuel_encoded = fuel_encoder.transform([fuel])[0]
            car_age = 2025 - year
            features = [[brand_encoded, car_age, km_driven, fuel_encoded]]

            if model_choice == "Linear Regression":
                prediction = model_lr.predict(features)[0]
                prediction = max(0, int(prediction))  # 🔧 This line ensures no negative values
                st.success(f"💸 Predicted Price (Linear Regression): ₹ {prediction:,}")

            else:
                prediction = model_rf.predict(features)[0]
                prediction = max(0, int(prediction))  # Prevents negative values
                st.success(f"💸 Predicted Price (Random Forest): ₹ {prediction:,}")

        except Exception as e:
            st.error(f"❌ Prediction failed: {e}")
