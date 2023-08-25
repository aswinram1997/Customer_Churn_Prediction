import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model


# Load the model and preprocessing steps
model = load_model("model/my_model.h5")
preprocessing_steps = joblib.load(open("model/preprocessing_steps.pkl", "rb"))


# Function to preprocess input data
def preprocess_input(input_data):
    # Apply preprocessing steps here
    # Apply the same preprocessing steps as used during model training
    X_input = input_data.copy()  # Make a copy of input_data for preprocessing

    # Apply the same preprocessing steps as used during model training
    num_imputer = preprocessing_steps['num_imputer']
    scaler = preprocessing_steps['scaler']
    encoder = preprocessing_steps['encoder']

    # Data cleaning and imputation
    X_input_numerical = X_input.select_dtypes(include=['float64', 'int64'])
    X_input_imputed = pd.DataFrame(num_imputer.transform(X_input_numerical), columns=X_input_numerical.columns)

    # Feature scaling
    X_input_scaled = pd.DataFrame(scaler.transform(X_input_imputed), columns=X_input_imputed.columns)

    # Feature encoding
    X_input_categorical = X_input.select_dtypes(include=['object'])
    X_input_encoded = pd.DataFrame(encoder.transform(X_input_categorical).toarray(), columns=encoder.get_feature_names_out(X_input_categorical.columns))

    # Concatenate scaled numerical features and encoded categorical features
    processed_data = pd.concat([X_input_scaled, X_input_encoded], axis=1)
    
    return processed_data


def make_predictions(input_data):
    processed_input = preprocess_input(input_data)
    predictions = model.predict(processed_input)
    
    return predictions


# Streamlit app code
def main():
    
    # Title
    st.header("Customer Churn Analytics")
  
    
    menu = ["Prediction", "Dashboard"]
    choice = st.sidebar.selectbox("Select Page", menu)

    if choice == "Prediction":
        
        # content
        st.write("The Prediction page allows users to input specific customer information, including details like tenure, preferred login device, and order count. Utilizing a pre-trained machine learning model, the app provides churn predictions for customers based on the entered data, offering insights into the likelihood of churn along with a model interpretation visual")
        
        # Create input fields for user to input data
        st.subheader("Enter Customer Information for Churn Prediction")
        tenure = st.slider("Tenure (months)", 0.0, 61.0, step=1.0)
        preferred_login_device = st.selectbox("Preferred Login Device", ["Mobile Phone", "Phone", "Computer"])
        city_tier = st.slider("City Tier", 1, 3, step=1)
        warehouse_to_home = st.slider("Distance from Warehouse to Home", 5.0, 127.0, step=1.0)
        preferred_payment_mode = st.selectbox("Preferred Payment Mode", ["Debit Card", "UPI", "CC", "Cash on Delivery", "E wallet", "COD", "Credit Card"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        hours_on_app = st.slider("Hour Spend On App", 0.0, 5.0, step=0.1)
        num_devices_registered = st.slider("Number of Devices Registered", 1, 6, step=1)
        preferred_order_cat = st.selectbox("Preferred Order Category", ["Laptop & Accessory", "Mobile", "Mobile Phone", "Others", "Fashion", "Grocery"])
        satisfaction_score = st.slider("Satisfaction Score", 1, 5, step=1)
        marital_status = st.selectbox("Marital Status", ["Single", "Divorced", "Married"])
        num_addresses = st.slider("Number of Addresses", 1, 22, step=1)
        complain = 1 if st.selectbox("Complain", ["No", "Yes"]) == "Yes" else 0
        order_amount_hike = st.slider("Order Amount Hike From Last Year (%)", 11.0, 26.0, step=0.1)
        coupon_used = st.slider("Number of Coupons Used", 0.0, 16.0, step=1.0)
        order_count = st.slider("Order Count", 1.0, 16.0, step=1.0)
        days_since_last_order = st.slider("Days Since Last Order", 0.0, 46.0, step=1.0)
        cashback_amount = st.slider("Cashback Amount", 0.0, 324.99, step=0.01)

        if st.button("Predict Churn"):
            input_data = pd.DataFrame({
            "Tenure": [tenure],
            "PreferredLoginDevice": [preferred_login_device],
            "CityTier": [city_tier],
            "WarehouseToHome": [warehouse_to_home],
            "PreferredPaymentMode": [preferred_payment_mode],
            "Gender": [gender],
            "HourSpendOnApp": [hours_on_app],
            "NumberOfDeviceRegistered": [num_devices_registered],
            "PreferedOrderCat": [preferred_order_cat],
            "SatisfactionScore": [satisfaction_score],
            "MaritalStatus": [marital_status],
            "NumberOfAddress": [num_addresses],
            "Complain": [complain],
            "OrderAmountHikeFromlastYear": [order_amount_hike],
            "CouponUsed": [coupon_used],
            "OrderCount": [order_count],
            "DaySinceLastOrder": [days_since_last_order],
            "CashbackAmount": [cashback_amount]
        })
            predictions = make_predictions(input_data)

            for prediction in predictions:
                churn_prob = prediction[0]
                churn_status = "Churn" if churn_prob >= 0.5 else "No Churn"
                st.write(f"The customer is likely to experience {churn_status} with a probability of {churn_prob:.2f}")
                
            st.markdown("## Model Interpretation")
            st.image("images/shap_plot.png", use_column_width=True)  # Display the SHAP plot image


    elif choice == "Dashboard":
        
        # content
        st.write("The Dashboard page provides a series of visualizations are presented, each shedding light on different aspects of customer churn. These visualizations encompass critical churn factors such as city tier, satisfaction score, and last order days. By offering a holistic view of various churn-related metrics, the dashboard empowers users to quickly grasp trends and correlations within the dataset.")
        
        # Create 3 columns to arrange the charts side by side
        col1, col2, col3 = st.columns(3)

        # List of image filenames and captions
        image_data = [
            {"file": "images/churn_chart.png", "caption": "Churn vs Not Churn"},
            {"file": "images/addresscount_chart.png", "caption": "Churn by Adress Count"},
            {"file": "images/cashback_chart.png", "caption": "Churn by Cashback"},
            {"file": "images/citytier_chart.png", "caption": "Churn by CityTier"},
            {"file": "images/complain_chart.png", "caption": "Churn by Complain"},
            {"file": "images/lastorder_chart.png", "caption": "Churn by Last Order"},
            {"file": "images/ordercat_chart.png", "caption": "Churn by Order Category"},
            {"file": "images/satisfaction_chart.png", "caption": "Churn by Satisfaction Score"},
            {"file": "images/tenure_chart.png", "caption": "Churn by Tenure"}
        ]

        # Loop through the image data and display images vertically
        for data in image_data:
            st.image(data["file"], caption=data["caption"], use_column_width=True)

            

if __name__ == "__main__":
    main()