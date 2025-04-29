# streamlit_app.py

import streamlit as st
import pandas as pd
import json
import os
import tensorflow as tf
import joblib

# --- Set page configuration
st.set_page_config(page_title="User Behavior & Traffic Prediction", page_icon="🚀", layout="wide")

# --- Load Models
@st.cache_resource
def load_models():
    model_m1 = tf.keras.models.load_model('M1_model.h5')
    model_m9 = tf.keras.models.load_model('M9_model.h5')
    traffic_model = joblib.load('TrafficPrediction.pkl')
    return model_m1, model_m9, traffic_model

model_m1, model_m9, traffic_model = load_models()

# --- User management
USERS_FILE = 'Users.json'

def load_users():
    if not os.path.exists(USERS_FILE):
        return {}
    with open(USERS_FILE, 'r') as f:
        return json.load(f)

def save_users(users):
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)

def signup(username, password):
    users = load_users()
    if username in users:
        return False
    users[username] = password
    save_users(users)
    return True

def login(username, password):
    users = load_users()
    return users.get(username) == password

# --- Initialize session state
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if 'page' not in st.session_state:
    st.session_state.page = 'login'

# --- Pages
if st.session_state.page == 'login':
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if login(username, password):
            st.session_state.logged_in = True
            st.success("Logged in successfully!")
            st.session_state.page = 'upload'
            st.experimental_rerun()
        else:
            st.error("Incorrect username or password.")

    if st.button("Create New Account"):
        st.session_state.page = 'signup'
        st.experimental_rerun()

elif st.session_state.page == 'signup':
    st.title("Create New Account")
    new_username = st.text_input("New Username")
    new_password = st.text_input("New Password", type="password")

    if st.button("Sign Up"):
        if signup(new_username, new_password):
            st.success("Account created successfully!")
            st.session_state.page = 'login'
            st.experimental_rerun()
        else:
            st.error("Username already exists. Try another one.")

    if st.button("Back to Login"):
        st.session_state.page = 'login'
        st.experimental_rerun()

elif st.session_state.logged_in and st.session_state.page == 'upload':
    st.title("Upload Files for Prediction")
    st.info("Please upload 3 files: Bundle 1, Bundle 2, and Traffic")

    uploaded_files = st.file_uploader("Upload your 3 files", accept_multiple_files=True, type=['csv'])

    if uploaded_files and len(uploaded_files) == 3:
        try:
            df1 = pd.read_csv(uploaded_files[0])
            df2 = pd.read_csv(uploaded_files[1])
            df3 = pd.read_csv(uploaded_files[2])

            # Predictions
            pred1 = model_m1.predict(df1)
            pred2 = model_m9.predict(df2)
            pred3 = traffic_model.predict(df3)

            # Average combination
            final_prediction = (pred1.flatten() + pred2.flatten() + pred3.flatten()) / 3

            # Display
            st.subheader("Predictions")
            prediction_df = pd.DataFrame({
                "Bundle 1 Prediction": pred1.flatten(),
                "Bundle 2 Prediction": pred2.flatten(),
                "Traffic Prediction": pred3.flatten(),
                "Final Combined Prediction": final_prediction
            })
            st.dataframe(prediction_df)

            # Download button
            csv = prediction_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Predictions as CSV", csv, "predictions.csv", "text/csv")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.warning("Please upload exactly 3 files.")
