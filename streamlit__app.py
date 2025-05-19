# streamlit_app.py

import streamlit as st
import pandas as pd
import json
import os

# Optional: Uncomment if you want to load models
# import tensorflow as tf
# import joblib

# --- Set page configuration
st.set_page_config(
    page_title="DataPred",
    page_icon="🚀",
    layout="wide"
)

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

# --- Page navigation and logic
def show_login():
    st.title("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if login(username, password):
            st.session_state.logged_in = True
            st.session_state.page = 'upload'
            st.success("Logged in successfully!")
            st.experimental_rerun()
        else:
            st.error("Incorrect username or password.")

    if st.button("Create New Account"):
        st.session_state.page = 'signup'
        st.experimental_rerun()

def show_signup():
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

def show_upload():
    st.title("Upload Files for Prediction")
    st.info("Upload any number of files (any type). CSV, Excel, text, and images are previewed below.")

    uploaded_files = st.file_uploader("Upload your files", accept_multiple_files=True)

    if uploaded_files:
        for idx, file in enumerate(uploaded_files):
            st.write(f"---\n**File {idx + 1}: {file.name}**")
            st.write(f"File type: {file.type}")

            try:
                if file.name.endswith('.csv'):
                    df = pd.read_csv(file)
                    st.dataframe(df.head())
                elif file.name.endswith('.xlsx'):
                    df = pd.read_excel(file)
                    st.dataframe(df.head())
                elif file.name.endswith('.txt'):
                    content = file.read().decode("utf-8")
                    st.text(content[:500])  # Show first 500 chars
                elif file.type.startswith("image/"):
                    st.image(file)
                else:
                    st.write("File preview not supported for this type.")
            except Exception as e:
                st.warning(f"Could not preview file '{file.name}': {e}")

        # Add your prediction/model logic here as needed

    else:
        st.info("No files uploaded yet.")

    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.page = 'login'
        st.experimental_rerun()

# --- Main app logic
if st.session_state.page == 'login':
    show_login()
elif st.session_state.page == 'signup':
    show_signup()
elif st.session_state.logged_in and st.session_state.page == 'upload':
    show_upload()
else:
    st.session_state.page = 'login'
    st.experimental_rerun()