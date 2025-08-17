import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler

# --- 1. LOAD SAVED MODEL AND SCALER ---
# Using a function with st.cache_resource to load these only once
@st.cache_resource
def load_assets():
    """Loads the trained model and scaler from disk."""
    try:
        model = tf.keras.models.load_model('regression_model.h5')
        with open('scaler1.pkl', 'rb') as file:
            scaler = pickle.load(file)
        return model, scaler
    except FileNotFoundError:
        st.error("Error: Model ('regression_model.h5') or scaler ('scaler1.pkl') not found.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading the files: {e}")
        st.stop()

model, scaler = load_assets()

# --- 2. STREAMLIT APP UI ---
st.set_page_config(layout="wide")
st.title('🏦 Customer Salary Estimator')
st.write("This app predicts the estimated salary of a bank customer based on their details. Please provide the customer's information in the sidebar.")

# --- 3. SIDEBAR FOR USER INPUT ---
st.sidebar.header('Customer Details')

def get_user_input():
    """Creates sidebar widgets and returns user inputs as a DataFrame."""
    credit_score = st.sidebar.slider('Credit Score', 300, 850, 650)
    geography = st.sidebar.selectbox('Geography', ('France', 'Germany', 'Spain'))
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female'))
    age = st.sidebar.slider('Age', 18, 100, 38)
    tenure = st.sidebar.slider('Tenure (years)', 0, 10, 5)
    balance = st.sidebar.number_input('Balance', 0.0, 250000.0, 75000.0, step=1000.0)
    num_of_products = st.sidebar.slider('Number of Products', 1, 4, 1)
    has_cr_card = st.sidebar.selectbox('Has Credit Card?', ('Yes', 'No'))
    is_active_member = st.sidebar.selectbox('Is Active Member?', ('Yes', 'No'))
    exited = st.sidebar.selectbox('Has the customer exited?', ('Yes', 'No'))

    # Create a dictionary from the inputs
    input_dict = {
        'CreditScore': credit_score,
        'Geography': geography,
        'Gender': gender,
        'Age': age,
        'Tenure': tenure,
        'Balance': balance,
        'NumOfProducts': num_of_products,
        'HasCrCard': 1 if has_cr_card == 'Yes' else 0,
        'IsActiveMember': 1 if is_active_member == 'Yes' else 0,
        'Exited': 1 if exited == 'Yes' else 0
    }
    
    return pd.DataFrame([input_dict])

input_df = get_user_input()

# --- 4. PREPROCESSING AND PREDICTION ---
st.subheader('Customer Input:')
st.write(input_df)

# Create a button to trigger the prediction
if st.button('Predict Estimated Salary', key='predict_button'):
    
    # --- Preprocessing Steps ---
    # This must match the preprocessing from your notebook
    
    # a. Encode Gender
    # Note: Your original notebook saved an unfitted LabelEncoder.
    # We will manually encode it here based on common practice (Female=0, Male=1)
    processed_df = input_df.copy()
    processed_df['Gender'] = processed_df['Gender'].apply(lambda x: 1 if x == 'Male' else 0)

    # b. One-Hot Encode Geography
    # Note: Your original notebook saved an unfitted OneHotEncoder.
    # We will manually encode it here.
    geo_dummies = pd.get_dummies(processed_df['Geography'], prefix='Geography').reindex(columns=['Geography_France', 'Geography_Germany', 'Geography_Spain'], fill_value=0)
    processed_df = pd.concat([processed_df, geo_dummies], axis=1)
    processed_df.drop('Geography', axis=1, inplace=True)

    # c. Ensure column order is the same as in training
    # This is a critical step!
    expected_columns = [
        'CreditScore', 'Gender', 'Age', 'Tenure', 'Balance', 
        'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'Exited',
        'Geography_France', 'Geography_Germany', 'Geography_Spain'
    ]
    processed_df = processed_df.reindex(columns=expected_columns)

    # d. Scale the features
    try:
        scaled_features = scaler.transform(processed_df)
        
        # --- Make Prediction ---
        prediction = model.predict(scaled_features)
        predicted_salary = prediction[0][0]

        # --- Display Result ---
        st.success(f"**Predicted Estimated Salary: ${predicted_salary:,.2f}**")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.error("Please ensure the input values are correct and the saved 'scaler1.pkl' is valid.")

