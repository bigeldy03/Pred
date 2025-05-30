import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Bundle Forecast App", layout="wide")

# --- Auth Setup ---
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

name, authentication_status, username = authenticator.login("Login", "main")

if authentication_status is False:
    st.error("Username/password is incorrect")
elif authentication_status is None:
    st.warning("Please enter your username and password")
elif authentication_status:
    st.sidebar.title(f"Welcome {name}")
    authenticator.logout("Logout", "sidebar")

    st.markdown("""
        <h1 style='text-align: center; color: #4CAF50;'>📊📶 Bundle & Traffic Predictor</h1>
    """, unsafe_allow_html=True)

    st.markdown("---")

    uploaded_bundle_file = st.file_uploader("📂 Upload your <b>bundle usage</b> data", type=['csv', 'xlsx'], key="bundle")
    uploaded_traffic_file = st.file_uploader("📂 Upload your <b>traffic</b> data", type=['csv', 'xlsx'], key="traffic")

    def process_dates(df):
        df['Date'] = pd.to_numeric(df['Date'], errors='coerce')
        df = df[(df['Date'] > 0) & (df['Date'] < 50000)]
        df['Date'] = pd.to_datetime(df['Date'], origin='1899-12-30', unit='D')
        return df

    def preprocess(df):
        df = df.fillna(df.mean(numeric_only=True))
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'Date' in numerical_cols:
            numerical_cols.remove('Date')
        scaler = MinMaxScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        return df, scaler, numerical_cols

    def create_sequences(data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data.iloc[i:i + seq_length].values
            y = data.iloc[i + seq_length].values
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def build_lstm_model(input_shape, output_dim):
        model = Sequential()
        model.add(LSTM(64, activation='relu', input_shape=input_shape))
        model.add(Dense(output_dim))
        model.compile(optimizer='adam', loss='mse')
        return model

    def predict_with_lstm(df, seq_length=10):
        df, scaler, numerical_cols = preprocess(df)
        X, y = create_sequences(df[numerical_cols], seq_length)
        X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
        model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]), output_dim=y.shape[1])
        model.fit(X_train, y_train, epochs=20, verbose=0)
        preds = model.predict(X_test)
        return preds, y_test, numerical_cols, scaler, model

    if uploaded_bundle_file and uploaded_traffic_file:
        try:
            bundle_df = pd.read_csv(uploaded_bundle_file) if uploaded_bundle_file.name.endswith('.csv') else pd.read_excel(uploaded_bundle_file)
            traffic_df = pd.read_csv(uploaded_traffic_file) if uploaded_traffic_file.name.endswith('.csv') else pd.read_excel(uploaded_traffic_file)

            st.success("✅ Files uploaded successfully")
            bundle_df = process_dates(bundle_df)
            traffic_df = process_dates(traffic_df)

            bundle_preds, _, bundle_cols, _, _ = predict_with_lstm(bundle_df)
            traffic_preds, _, traffic_cols, _, _ = predict_with_lstm(traffic_df)

            trends = {col: bundle_preds[:, i].mean() for i, col in enumerate(bundle_cols)}
            best_bundle = max(trends, key=trends.get)
            best_index = bundle_cols.index(best_bundle)
            traffic_col = [col for col in traffic_cols if best_bundle.lower() in col.lower()]
            traffic_index = traffic_cols.index(traffic_col[0]) if traffic_col else 0

            st.balloons()
            st.markdown(f"""
                <div style='background-color:#e0f7fa;padding:20px;border-radius:10px'>
                <h2 style='text-align:center;color:#00796b;'>🌟 Recommended Bundle: <span style='color:#e91e63'>{best_bundle}</span></h2>
                <p style='text-align:center;'>Based on current trends, this bundle shows the strongest growth potential.</p>
                </div>
            """, unsafe_allow_html=True)

            # --- Plot Bundle Forecast ---
            fig1, ax1 = plt.subplots()
            sns.lineplot(data=bundle_preds[:, best_index], ax=ax1, color="blue")
            ax1.set_title(f"📈 Forecast for {best_bundle}")
            st.pyplot(fig1)

            # --- Plot Traffic Forecast ---
            fig2, ax2 = plt.subplots()
            sns.lineplot(data=traffic_preds[:, traffic_index], ax=ax2, color="orange")
            ax2.set_title(f"📊 Predicted Traffic for {best_bundle}")
            st.pyplot(fig2)

        except Exception as e:
            st.error(f"❌ Error during processing: {e}")
    else:
        st.info("👆 Please upload both bundle and traffic files to proceed.")
