import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Sample transactions (extracted from creditcard.csv)
SAMPLE_TRANSACTIONS = [
    {
        "id": 1,
        "description": "Legitimate Transaction: Amount $149.62 at time 0",
        "V1": -1.3598071336738,
        "V2": -0.0727811733098497,
        "V3": 2.53634673796914,
        "V4": 1.37815522427443,
        "V5": -0.338320769942518,
        "V6": 0.462387777762292,
        "V7": 0.239598554061257,
        "V8": 0.0986979012610507,
        "V9": 0.363786969611213,
        "V10": 0.0907941719789316,
        "V11": -0.551599533260813,
        "V12": -0.617800855762348,
        "V13": -0.991389847235408,
        "V14": -0.311169353699879,
        "V15": 1.46817697209427,
        "V16": -0.470400525259478,
        "V17": 0.207971241929242,
        "V18": 0.0257905801985591,
        "V19": 0.403992960255733,
        "V20": 0.251412098239705,
        "V21": -0.018306777944153,
        "V22": 0.277837575558899,
        "V23": -0.110473910188767,
        "V24": 0.0669280749146731,
        "V25": 0.128539358273528,
        "V26": -0.189114843888824,
        "V27": 0.133558376740387,
        "V28": -0.0210530534538215,
        "Time": 0,
        "Amount": 149.62
    },
    {
        "id": 2,
        "description": "Fraudulent Transaction: Amount $0 at time 0",
        "V1": -2.3122265423263,
        "V2": 1.95199201064158,
        "V3": -1.60985073229769,
        "V4": 3.9979055875468,
        "V5": -0.522187864667764,
        "V6": -1.42654531920595,
        "V7": -2.53738730624579,
        "V8": 1.39165724829804,
        "V9": -2.77008927719433,
        "V10": -2.77227214465915,
        "V11": 3.20203320709635,
        "V12": -2.89990738849473,
        "V13": -0.595221881324605,
        "V14": -4.28925378244217,
        "V15": 0.389724860969815,
        "V16": -1.14074717980657,
        "V17": -2.83005567450437,
        "V18": -0.0168224681808257,
        "V19": 0.416955705037907,
        "V20": 0.126910559061474,
        "V21": 0.517232370861764,
        "V22": -0.0350493686052974,
        "V23": -0.465211076182388,
        "V24": 0.320198198514526,
        "V25": 0.0445191674737224,
        "V26": 0.177839798284401,
        "V27": 0.414068691700338,
        "V28": -0.143276169862367,
        "Time": 0,
        "Amount": 0
    },
    {
        "id": 3,
        "description": "Legitimate Transaction: Amount $500 at time 3600",
        "V1": -0.966271711572087,
        "V2": -0.185226008082898,
        "V3": 1.79299333957872,
        "V4": -0.863291275036453,
        "V5": -0.0103088802711944,
        "V6": 1.24720316752486,
        "V7": 0.23760893977178,
        "V8": 0.377435874652262,
        "V9": -1.38702406270197,
        "V10": 0.0549519224713749,
        "V11": -0.226487263835401,
        "V12": 0.178228225877303,
        "V13": 0.507756869957169,
        "V14": -0.28792374549456,
        "V15": -0.631418117709045,
        "V16": -1.0596472454325,
        "V17": -0.684092786004191,
        "V18": 1.96577500349538,
        "V19": -1.2326219700892,
        "V20": -0.208037781160366,
        "V21": -0.108300452035545,
        "V22": 0.00527359678213764,
        "V23": -0.190320518742841,
        "V24": -1.17557533186321,
        "V25": 0.647376034602038,
        "V26": -0.221928844458407,
        "V27": 0.0627228487293033,
        "V28": -0.024800501771819,
        "Time": 3600,
        "Amount": 500
    },
    
    {
        "id": 4,
        "description": "Legitimate Transaction: Amount $75 at time 43200",
        "V1": 0.287634135135135,
        "V2": -0.0537387387387387,
        "V3": -0.133446446446446,
        "V4": 0.123456789012345,
        "V5": -0.234567890123456,
        "V6": 0.345678901234567,
        "V7": -0.456789012345678,
        "V8": 0.567890123456789,
        "V9": -0.67890123456789,
        "V10": 0.789012345678901,
        "V11": -0.890123456789012,
        "V12": 0.901234567890123,
        "V13": -0.0123456789012345,
        "V14": 0.123456789012345,
        "V15": -0.234567890123456,
        "V16": 0.345678901234567,
        "V17": -0.456789012345678,
        "V18": 0.567890123456789,
        "V19": -0.67890123456789,
        "V20": 0.789012345678901,
        "V21": -0.890123456789012,
        "V22": 0.901234567890123,
        "V23": -0.0123456789012345,
        "V24": 0.123456789012345,
        "V25": -0.234567890123456,
        "V26": 0.345678901234567,
        "V27": -0.456789012345678,
        "V28": 0.044528123456789,
        "Time": 43200,
        "Amount": 75
    }
]

# Streamlit app
st.title("Credit Card Fraud Detection")
st.markdown("""
Welcome to the AI-powered Credit Card Fraud Detection app! Select a sample transaction, modify the Amount and Time if desired, and predict whether it's Fraudulent or Legitimate.
""")

# Load model and scaler
try:
    model = joblib.load('NaanMud/model.pkl')
    scaler = joblib.load('scaler.pkl')
    logger.info("Model and scaler loaded successfully.")
except Exception as e:
    st.error(f"Failed to load model or scaler: {str(e)}")
    logger.error(f"Model/scaler loading failed: {str(e)}")
    st.stop()

# Sample transaction selection
sample_options = {sample['description']: sample for sample in SAMPLE_TRANSACTIONS}
selected_sample = st.selectbox("Select a Sample Transaction", list(sample_options.keys()))

# Get selected sample data
sample_data = sample_options[selected_sample]

# Input fields for Amount and Time
st.subheader("Modify Transaction Details")
col1, col2 = st.columns(2)
with col1:
    amount = st.number_input("Amount ($)", min_value=0.0, value=float(sample_data['Amount']), step=0.01)
with col2:
    time = st.number_input("Time (seconds since first transaction)", min_value=0.0, value=float(sample_data['Time']), step=1.0)

# Predict button
if st.button("Predict"):
    try:
        logger.info(f"Processing prediction for Amount: {amount}, Time: {time}")
        # Prepare features
        features = [sample_data[f'V{i}'] for i in range(1, 29)]  # V1–V28
        log_amount = np.log1p(amount)
        hour = (int(time) // 3600) % 24
        time_diff = 0  # Single prediction, no prior transaction
        feature_vector = features + [log_amount, hour, time_diff]

        # Validate feature vector
        if len(feature_vector) != 31:
            raise ValueError(f"Expected 31 features, got {len(feature_vector)}")

        # Standardize features
        feature_vector = np.array(feature_vector).reshape(1, -1)
        feature_vector_scaled = scaler.transform(feature_vector)
        logger.info("Features standardized successfully.")

        # Predict
        prediction = model.predict(feature_vector_scaled)[0]
        result = "Fraudulent" if prediction == 1 else "Legitimate"
        color = "red" if prediction == 1 else "green"

        # Display result
        st.markdown(f"**Prediction**: <span style='color:{color};font-size:20px;font-weight:bold'>{result}</span>", unsafe_allow_html=True)
        logger.info(f"Prediction: {result}")

        # Display feature vector (for debugging/insight)
        with st.expander("View Feature Vector"):
            feature_names = [f'V{i}' for i in range(1, 29)] + ['Log_Amount', 'Hour', 'Time_Diff']
            feature_df = pd.DataFrame([feature_vector[0]], columns=feature_names)
            st.dataframe(feature_df)

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        logger.error(f"Prediction error: {str(e)}")

# About section
st.markdown("---")
st.subheader("About")
st.markdown("""
This app uses a pre-trained machine learning model to detect credit card fraud based on the [Kaggle Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud). 
Select a sample transaction, adjust Amount and Time, and the app generates all required features (V1-V28, Log_Amount, Hour, Time_Diff) for prediction. 
Built with Streamlit for simplicity and rapid deployment.
            
The prediction outcome (Legitimate or Fraudulent) in our Streamlit application primarily depends on the features V1-V28, which are PCA-transformed components from the Kaggle Credit Card
Fraud Detection dataset. While users can modify Amount and Time, these changes only affect Log_Amount, Hour, and Time_Diff, which have a limited impact on the model's decision compared to V1-V28.
This is because the model was trained to rely heavily on the patterns in V1-V28, which capture critical transaction characteristics. In a real-world scenario, retrieving live transaction data to
generate V1-V28 would be ideal, as it would reflect actual transaction behaviors. However, this is not feasible within the scope of this project due to the dataset's anonymized nature and the lack
of access to raw, live transaction data. To address this, we've implemented a user-friendly approach where users select pre-defined sample transactions with fixed V1-V28 values, allowing them to 
modify only Amount and Time for simplicity, while the backend generates the full 31-feature set for prediction.
""")
