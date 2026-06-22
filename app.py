import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Page Configuration
# -------------------------------

st.set_page_config(
    page_title="California House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# -------------------------------
# Load Model
# -------------------------------

saved_model = joblib.load("house_price_model.pkl")

model = saved_model["model"]

# -------------------------------
# Sidebar
# -------------------------------

st.sidebar.title("📊 Project Information")

st.sidebar.info(
    """
    California House Price Prediction using Machine Learning

    Model Used:
    Gradient Boosting Regressor

    Performance:
    • Train R² = 0.8049
    • Test R² = 0.7756

    Developed using:
    • Python
    • Scikit-Learn
    • Streamlit
    """
)

# -------------------------------
# Main Title
# -------------------------------

st.title("🏠 California House Price Prediction")

st.markdown(
    "Predict California house prices using a trained Machine Learning model."
)

st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Model", "Gradient Boosting")

with col2:
    st.metric("Test R²", "0.776")

with col3:
    st.metric("Features", "8")

st.info(
    """
    Dataset: California Housing Dataset (Scikit-Learn)

    This application predicts median house values based on demographic and housing-related features.
    """
)

# -------------------------------
# Input Layout
# -------------------------------

col1, col2 = st.columns(2)

with col1:

    medinc = st.number_input(
        "Median Income",
        min_value=0.0,
        value=3.5
    )

    houseage = st.number_input(
        "House Age",
        min_value=0.0,
        value=25.0
    )

    averooms = st.number_input(
        "Average Rooms",
        min_value=0.0,
        value=5.0
    )

    avebedrms = st.number_input(
        "Average Bedrooms",
        min_value=0.0,
        value=1.0
    )

with col2:

    population = st.number_input(
        "Population",
        min_value=0.0,
        value=1000.0
    )

    aveoccup = st.number_input(
        "Average Occupancy",
        min_value=0.0,
        value=3.0
    )

    latitude = st.number_input(
        "Latitude",
        value=34.0
    )

    longitude = st.number_input(
        "Longitude",
        value=-118.0
    )

# -------------------------------
# Prediction Button
# -------------------------------

if st.button("🔮 Predict House Price"):

    input_df = pd.DataFrame({
        "medinc": [medinc],
        "houseage": [houseage],
        "averooms": [averooms],
        "avebedrms": [avebedrms],
        "population": [population],
        "aveoccup": [aveoccup],
        "latitude": [latitude],
        "longitude": [longitude]
    })

    prediction = model.predict(input_df)[0]

    # Dataset target is in units of $100,000
    price_usd = prediction * 100000

    # Approximate conversion
    price_inr = price_usd * 83

    st.success("✅ Prediction Generated Successfully")

    st.metric(
        label="Estimated House Price (USD)",
        value=f"${price_usd:,.2f}"
    )

    st.metric(
        label="Estimated House Price (INR)",
        value=f"₹{price_inr:,.0f}"
    )

# -------------------------------
# Footer
# -------------------------------

st.markdown("---")

st.caption(
    "Built with Streamlit | California Housing Dataset | Gradient Boosting Regressor"
)