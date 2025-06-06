import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
import seaborn as sns

# --- Streamlit UI Setup ---
st.set_page_config(page_title="Bundle Forecast App", layout="wide")
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>📊📶 Bundle & Traffic Predictor</h1>
""", unsafe_allow_html=True)
st.markdown("---")

uploaded_bundle_file = st.file_uploader("📂 Upload your <b>bundle usage</b> data", type=['csv', 'xlsx'], key="bundle")
uploaded_traffic_file = st.file_uploader("📂 Upload your <b>traffic</b> data", type=['csv', 'xlsx'], key="traffic")

# --- Fix Excel-style numeric date conversion ---
def infer_and_process_date(df):
    for col in df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            try:
                if np.issubdtype(df[col].dtype, np.number):
                    df[col] = pd.to_datetime(df[col], origin='1899-12-30', unit='D', errors='coerce')
                else:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                df = df.dropna(subset=[col])
                df = df.sort_values(by=col)
                return df, col
            except:
                continue
    return df, None

# --- Preprocess numerical data ---
def preprocess(df):
    df = df.fillna(df.mean(numeric_only=True))
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    scaler = MinMaxScaler()
    df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    return df, scaler, numerical_cols

# --- Sequence creation for LSTM ---
def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data.iloc[i:i + seq_length].values
        y = data.iloc[i + seq_length].values
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# --- LSTM model builder ---
def build_lstm_model(input_shape, output_dim):
    model = Sequential()
    model.add(LSTM(64, activation='relu', input_shape=input_shape))
    model.add(Dense(output_dim))
    model.compile(optimizer='adam', loss='mse')
    return model

# --- Prediction function ---
def predict_with_lstm(df, seq_length=10):
    df, scaler, numerical_cols = preprocess(df)
    X, y = create_sequences(df[numerical_cols], seq_length)
    if len(X) == 0:
        raise ValueError("Not enough data to create sequences.")
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)
    model = build_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]), output_dim=y.shape[1])
    model.fit(X_train, y_train, epochs=20, verbose=0)
    preds = model.predict(X_test)
    return preds, y_test, numerical_cols, scaler, model

# --- Main Logic ---
if uploaded_bundle_file and uploaded_traffic_file:
    try:
        bundle_df = pd.read_csv(uploaded_bundle_file) if uploaded_bundle_file.name.endswith('.csv') else pd.read_excel(uploaded_bundle_file)
        traffic_df = pd.read_csv(uploaded_traffic_file) if uploaded_traffic_file.name.endswith('.csv') else pd.read_excel(uploaded_traffic_file)

        st.success("✅ Files uploaded successfully")

        bundle_df, bundle_date_col = infer_and_process_date(bundle_df)
        traffic_df, traffic_date_col = infer_and_process_date(traffic_df)

        st.subheader("📄 Preview of Bundle Data")
        st.dataframe(bundle_df.head())

        st.subheader("📄 Preview of Traffic Data")
        st.dataframe(traffic_df.head())

        # --- Run LSTM predictions ---
        bundle_preds, _, bundle_cols, _, _ = predict_with_lstm(bundle_df)
        traffic_preds, _, traffic_cols, _, _ = predict_with_lstm(traffic_df)

        trends = {col: bundle_preds[:, i].mean() for i, col in enumerate(bundle_cols)}
        best_bundle = max(trends, key=trends.get)
        best_index = bundle_cols.index(best_bundle)

        # Try match bundle name to traffic column
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
        ax1.set_xlabel("Days Ahead")
        ax1.set_ylabel("Normalized Bundle Usage")
        st.pyplot(fig1)

        # --- Plot Traffic Forecast ---
        fig2, ax2 = plt.subplots()
        sns.lineplot(data=traffic_preds[:, traffic_index], ax=ax2, color="orange")
        ax2.set_title(f"📊 Predicted Traffic for {best_bundle}")
        ax2.set_xlabel("Days Ahead")
        ax2.set_ylabel("Normalized Traffic")
        st.pyplot(fig2)

    except Exception as e:
        st.error(f"❌ Error during processing: {e}")
else:
    st.info("👆 Please upload both bundle and traffic files to proceed.")
