import streamlit as st
import pandas as pd
import pickle

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Delivery Time Prediction",
    layout="centered"
)

st.title("🚚 Delivery Time Prediction")
st.caption("Polynomial Regression Model")

# ---------------- Load Pickle ----------------
with open("delivery_time.pkl", "rb") as f:
    data = pickle.load(f)

model = data["model"]
poly = data["poly"]
encoders = data["label_encoders"]

st.markdown("---")

# ---------------- Input Layout ----------------
col1, col2 = st.columns(2)

with col1:
    order_id = st.number_input(
        "Order ID",
        min_value=1,
        step=1
    )

    distance_km = st.number_input(
        "Distance (km)",
        min_value=0.0,
        step=0.1
    )

    weather = st.selectbox(
        "Weather",
        encoders["Weather"].classes_.tolist()
    )

    traffic = st.selectbox(
        "Traffic Level",
        encoders["Traffic_Level"].classes_.tolist()
    )

with col2:
    time_of_day = st.selectbox(
        "Time of Day",
        encoders["Time_of_Day"].classes_.tolist()
    )

    vehicle = st.selectbox(
        "Vehicle Type",
        encoders["Vehicle_Type"].classes_.tolist()
    )

    prep_time = st.number_input(
        "Preparation Time (minutes)",
        min_value=0,
        step=1
    )

    experience = st.number_input(
        "Courier Experience (years)",
        min_value=0,
        step=1
    )

st.markdown("---")

# ---------------- Prediction ----------------
if st.button("Predict Delivery Time"):
    try:
        # Create input DataFrame
        input_df = pd.DataFrame({
            "Order_ID": [order_id],
            "Distance_km": [distance_km],
            "Weather": [weather],
            "Traffic_Level": [traffic],
            "Time_of_Day": [time_of_day],
            "Vehicle_Type": [vehicle],
            "Preparation_Time_min": [prep_time],
            "Courier_Experience_yrs": [experience]
        })

        # Encode categorical columns (PROPER WAY)
        input_df["Weather"] = encoders["Weather"].transform(input_df["Weather"])
        input_df["Traffic_Level"] = encoders["Traffic_Level"].transform(input_df["Traffic_Level"])
        input_df["Time_of_Day"] = encoders["Time_of_Day"].transform(input_df["Time_of_Day"])
        input_df["Vehicle_Type"] = encoders["Vehicle_Type"].transform(input_df["Vehicle_Type"])

        # Polynomial transform
        input_poly = poly.transform(input_df)

        # Prediction
        prediction = model.predict(input_poly)[0]

        st.success(
            f"⏱️ Estimated Delivery Time: **{round(prediction, 2)} minutes**"
        )

    except Exception as e:
        st.error(f"Error: {e}")
