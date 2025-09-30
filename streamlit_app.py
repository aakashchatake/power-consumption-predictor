import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import pandas as pd

# Load model and scalers with error handling
try:
    model = load_model("lstm_power_model.keras")
    scaler_X = joblib.load("scaler_X (2).pkl")
    scaler_y = joblib.load("scaler_y (2).pkl")
except Exception as e:
    st.error(f"Error loading model files: {str(e)}")
    st.error("Please ensure all model files are uploaded to the repository:")
    st.error("- lstm_power_model.keras")
    st.error("- scaler_X (2).pkl") 
    st.error("- scaler_y (2).pkl")
    st.stop()

# Load dataset for About Data tab with error handling
try:
    data = pd.read_excel("Cleaned_Dataset.xlsx")
    data['Datetime'] = pd.to_datetime(data['Datetime'], errors='coerce')
    data.dropna(subset=['Datetime'], inplace=True)
    data.set_index('Datetime', inplace=True)
except Exception as e:
    st.error(f"Error loading dataset: {str(e)}")
    st.error("Please ensure Cleaned_Dataset.xlsx is uploaded to the repository")
    data = None

# Page setup
st.set_page_config(page_title="⚡ Power Predictor", page_icon="⚡", layout="wide")

# Style
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #e0f7fa, #fce4ec);
}
.stButton>button {
    background-color: #ffcc00;
    color: black;
    font-size: 16px;
    font-weight: bold;
    border-radius: 10px;
}
</style>
""", unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["🔮 Predict", "📊 About Data", "⚙️ Model Info"])

# ====================== PREDICT TAB ======================
with tab1:
    st.title("⚡ Household Power Consumption Predictor")
    st.write("Adjust the sliders to simulate household conditions:")

    col1, col2, col3 = st.columns(3)
    with col1:
        reactive_power = st.slider("Global Reactive Power", 0.0, 2.0, 0.5)
    with col2:
        voltage = st.slider("Voltage (V)", 200.0, 250.0, 230.0)
    with col3:
        intensity = st.slider("Global Intensity (A)", 0.0, 50.0, 10.0)

    col4, col5, col6 = st.columns(3)
    with col4:
        sub1 = st.slider("Sub Metering 1", 0.0, 50.0, 0.0)
    with col5:
        sub2 = st.slider("Sub Metering 2", 0.0, 50.0, 0.0)
    with col6:
        sub3 = st.slider("Sub Metering 3", 0.0, 50.0, 0.0)

    if st.button("⚡ Predict"):
        with st.spinner("Calculating prediction... 🔄"):
            user_input = np.array([[reactive_power, voltage, intensity, sub1, sub2, sub3]])
            user_input_scaled = scaler_X.transform(user_input)
            # Tile input to (1, 24, 6) shape for LSTM trained on 24 timesteps
            user_input_scaled = np.tile(user_input_scaled, (1, 24, 1)).reshape(1, 24, 6)

            y_pred_scaled = model.predict(user_input_scaled)
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1))

        st.success(f"⚡ Predicted Power Consumption: {y_pred[0][0]:.2f} kW ⚡")
        st.balloons()

# ====================== ABOUT DATA TAB ======================
with tab2:
    st.header("📊 About Dataset")
    if data is not None:
        st.write(f"The dataset contains **{data.shape[0]} rows** and **{data.shape[1]} columns**.")
    else:
        st.error("Dataset not available - please check file upload")

    with st.expander("Column Information & Meaning"):
        st.write("""
        - **Global_active_power**: Total active power consumed (kW)  
        - **Global_reactive_power**: Reactive power consumed (kVar)  
        - **Voltage**: Voltage level (V)  
        - **Global_intensity**: Current intensity (A)  
        - **Sub_metering_1**: Energy for kitchen (Wh)  
        - **Sub_metering_2**: Energy for laundry (Wh)  
        - **Sub_metering_3**: Energy for AC/heating (Wh)  
        """)

    if data is not None:
        with st.expander("Column Relationships / Correlation"):
            st.write("Correlation matrix to show relations between features:")
            corr = data.corr()
            st.dataframe(corr.style.background_gradient(cmap='coolwarm'))

        with st.expander("Sample Data"):
            st.dataframe(data.sample(10))
    else:
        st.warning("Dataset features not available - upload Cleaned_Dataset.xlsx to view correlations and samples")

# ====================== MODEL INFO TAB ======================
with tab3:
    st.header("⚙️ Model Information")
    
    with st.expander("Architecture"):
        st.write("""
        - Model Type: LSTM (Long Short-Term Memory)  
        - Layers:  
            1. LSTM (50 units, return_sequences=True)  
            2. LSTM (50 units, return_sequences=False)  
            3. Dense (25 units)  
            4. Dense (1 unit, output)  
        - Activation: default (linear output for regression)  
        - Optimizer: Adam  
        - Loss: Mean Squared Error (MSE)
        """)

    with st.expander("How LSTM Works with This Dataset"):
        st.write("""
        - The LSTM model predicts **Global Active Power** based on past measurements of all features.  
        - Input sequence length = 24 hours (time_step=24).  
        - The model learns temporal patterns: how reactive power, voltage, intensity, and sub-metering affect future consumption.  
        - By sliding a window of 24 hours over the dataset, the model forecasts the next hour's active power.  
        """)

    with st.expander("Interactive Example Input → Prediction"):
        st.write("You can experiment with different input values below:")
        col1, col2, col3 = st.columns(3)
        with col1:
            r = st.number_input("Reactive Power", 0.0, 2.0, 0.5)
        with col2:
            v = st.number_input("Voltage", 200.0, 250.0, 230.0)
        with col3:
            i = st.number_input("Intensity", 0.0, 50.0, 10.0)
        col4, col5, col6 = st.columns(3)
        with col4:
            s1 = st.number_input("Sub1", 0.0, 50.0, 0.0)
        with col5:
            s2 = st.number_input("Sub2", 0.0, 50.0, 0.0)
        with col6:
            s3 = st.number_input("Sub3", 0.0, 50.0, 0.0)
        if st.button("Predict Example"):
            inp = np.array([[r,v,i,s1,s2,s3]])
            inp_scaled = scaler_X.transform(inp)
            # Tile input to (1, 24, 6) shape for LSTM trained on 24 timesteps
            inp_scaled = np.tile(inp_scaled, (1, 24, 1)).reshape(1, 24, 6)
            y_pred_scaled = model.predict(inp_scaled)
            y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1,1))
            st.success(f"Predicted Power: {y_pred[0][0]:.2f} kW ⚡")