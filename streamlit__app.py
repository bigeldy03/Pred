import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib  # For saving/loading models and scalers
from keras.models import Sequential
from keras.layers import LSTM, Dense
import xgboost as xgb
from prophet import Prophet

# --- Streamlit App ---
st.title("Time Series Prediction")

# --- Date Preprocessing (from your notebook) ---
def process_dates(df):
    df['Date'] = pd.to_numeric(df['Date'], errors='coerce')
    df = df[(df['Date'] > 0) & (df['Date'] < 50000)]
    df['Date'] = pd.to_datetime(df['Date'], origin='1899-12-30', unit='D')
    return df

# --- Numerical Data Preprocessing ---
def preprocess_numerical(df, target_column, scaler=None, is_training=True):
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Date' in numerical_cols:
        numerical_cols.remove('Date')
    if target_column in numerical_cols:
        numerical_cols.remove(target_column)

    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

    if is_training:
        scaler = MinMaxScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        return df, scaler
    else:
        df[numerical_cols] = scaler.transform(df[numerical_cols])
        return df, scaler

# --- Create Sequences for LSTM ---
def create_sequences(data, seq_length, target_column):
    xs = []
    ys = []
    target_index = data.columns.get_loc(target_column)
    for i in range(len(data) - seq_length):
        x = data.iloc[i:(i + seq_length)].drop(columns=[target_column])
        y = data.iloc[i + seq_length, target_index]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# --- LSTM Model ---
def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# --- XGBoost Model ---
def create_xgboost_model():
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    return model

# --- Prophet Model ---
def create_prophet_model():
    model = Prophet()
    return model

# --- Train and Evaluate ---
def train_and_evaluate(df, target_column, model_type="LSTM", seq_length=10):
    df = df.copy() # Avoid modifying original DataFrame
    df = process_dates(df)
    df, scaler = preprocess_numerical(df, target_column)

    if model_type == "LSTM":
        df_for_lstm = df.drop(columns=['Date'])
        X, y = create_sequences(df_for_lstm, seq_length, target_column)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
        model = create_lstm_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        model.fit(X_train, y_train, epochs=20, verbose=0)
        y_pred = model.predict(X_test)
    elif model_type == "XGBoost":
        df_for_xgboost = df.drop(columns=['Date'])
        X = df_for_xgboost.drop(columns=[target_column])
        y = df_for_xgboost[target_column]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
        model = create_xgboost_model()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
    elif model_type == "Prophet":
        df_for_prophet = df[['Date', target_column]].copy()
        df_for_prophet.columns = ['ds', 'y']  # Prophet requires 'ds' and 'y'
        model = create_prophet_model()
        model.fit(df_for_prophet)
        future = model.make_future_dataframe(periods=30)  # Predict 30 days into the future
        forecast = model.predict(future)
        y_pred = forecast.tail(len(df_for_prophet) - int(0.8 * len(df_for_prophet)))['yhat'].values
        y_test = df_for_prophet.tail(len(df_for_prophet) - int(0.8 * len(df_for_prophet)))['y'].values
    else:
        raise ValueError("Invalid model type")

    if model_type != "Prophet":
        y_pred = y_pred.flatten()
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return model, scaler, mse, mae, r2, y_test, y_pred

# --- Streamlit UI ---
uploaded_file = st.file_uploader("Upload your data (CSV or Excel)")
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file type")
            st.stop()

        st.subheader("Uploaded Data")
        st.dataframe(df.head())

        target_column = st.selectbox("Select the target column for prediction", df.columns)
        model_type = st.selectbox("Select the model", ["LSTM", "XGBoost", "Prophet"])
        seq_length = st.slider("LSTM Sequence Length", min_value=1, max_value=30, value=10) if model_type == "LSTM" else None

        if st.button("Train and Predict"):
            try:
                model, scaler, mse, mae, r2, y_test, y_pred = train_and_evaluate(df, target_column, model_type, seq_length)

                st.subheader("Evaluation Metrics")
                st.write(f"Mean Squared Error: {mse:.4f}")
                st.write(f"Mean Absolute Error: {mae:.4f}")
                st.write(f"R-squared: {r2:.4f}")

                st.subheader("Predictions vs. Actual")
                results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
                st.line_chart(results_df)

                # Save model and scaler (you might want to add a download button)
                joblib.dump(model, f'trained_{model_type}_model.joblib')
                if model_type != "Prophet":
                    joblib.dump(scaler, f'scaler.joblib')
                st.success(f"Trained {model_type} model and scaler saved.")

            except Exception as e:
                st.error(f"Error during training/prediction: {e}")

    except Exception as e:
        st.error(f"Error processing file: {e}")