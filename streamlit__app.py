import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
from keras.models import Sequential
from keras.layers import LSTM, Dense
import xgboost as xgb
from prophet import Prophet  # Import Prophet

# --- Streamlit App ---
st.title("Multivariate Time Series Prediction")

# --- Date Preprocessing ---
def process_dates(df):
    df['Date'] = pd.to_numeric(df['Date'], errors='coerce')
    df = df[(df['Date'] > 0) & (df['Date'] < 50000)]
    df['Date'] = pd.to_datetime(df['Date'], origin='1899-12-30', unit='D')
    return df

# --- Numerical Data Preprocessing ---
def preprocess_numerical(df, scaler=None, is_training=True):
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Date' in numerical_cols:
        numerical_cols.remove('Date')

    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].mean())

    if is_training:
        scaler = MinMaxScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        return df, scaler
    else:
        df[numerical_cols] = scaler.transform(df[numerical_cols])
        return df, scaler

# --- Create Sequences for LSTM (Multivariate) ---
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data.iloc[i:(i + seq_length)]
        y = data.iloc[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# --- LSTM Model (Multivariate Output) ---
def create_lstm_model(input_shape, output_dim):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape))
    model.add(Dense(output_dim))  # Output layer now has 'output_dim' units
    model.compile(optimizer='adam', loss='mse')
    return model

# --- XGBoost Model (Multivariate Output) ---
def create_xgboost_model(output_dim):
    model = xgb.XGBRegressor(
        objective='reg:squarederror',
        n_estimators=100,
        random_state=42
    )  # Basic XGBoost
    return model

# --- Prophet Model (Modified for Multiple Columns) ---
def create_prophet_models(df, numerical_cols):
    prophet_models = {}
    for col in numerical_cols:
        model = Prophet()
        prophet_models[col] = model
    return prophet_models

def train_prophet_models(prophet_models, df):
    prophet_forecasts = {}
    df_prophet = df[['Date'] + numerical_cols].copy()
    df_prophet = df_prophet.rename(columns={'Date': 'ds'})

    for col, model in prophet_models.items():
        temp_df = df_prophet[['ds', col]].rename(columns={col: 'y'})
        model.fit(temp_df)
        future = model.make_future_dataframe(periods=30, freq='D')  # Adjust periods as needed
        forecast = model.predict(future)
        prophet_forecasts[col] = forecast
    return prophet_forecasts

# --- Train and Evaluate ---
def train_and_evaluate(df, model_type="LSTM", seq_length=10):
    df = df.copy()
    df = process_dates(df)
    df, scaler = preprocess_numerical(df)  # Now preprocesses all numerical

    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if 'Date' in numerical_cols:
        numerical_cols.remove('Date')
    output_dim = len(numerical_cols)  # Number of columns to predict

    results = {}

    if model_type == "LSTM":
        X, y = create_sequences(df.drop(columns=['Date']), seq_length)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        model = create_lstm_model(
            input_shape=(X_train.shape[1], X_train.shape[2]), output_dim=output_dim
        )
        model.fit(X_train, y_train, epochs=20, verbose=0)
        y_pred = model.predict(X_test)
        y_test = y_test
        y_pred = y_pred

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        results = {
            "model": model,
            "scaler": scaler,
            "mse": mse,
            "mae": mae,
            "y_test": y_test,
            "y_pred": y_pred,
        }

    elif model_type == "XGBoost":
        X = df.drop(columns=['Date'] + numerical_cols)
        y = df[numerical_cols]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, shuffle=False
        )
        model = create_xgboost_model(output_dim=output_dim)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)

        results = {
            "model": model,
            "scaler": scaler,
            "mse": mse,
            "mae": mae,
            "y_test": y_test,
            "y_pred": y_pred,
        }

    elif model_type == "Prophet":
        prophet_models = create_prophet_models(df, numerical_cols)
        prophet_forecasts = train_prophet_models(prophet_models, df)

        # Basic evaluation (can be improved)
        total_mse = 0
        total_mae = 0
        for col in numerical_cols:
            forecast = prophet_forecasts[col]
            y_true = df.tail(len(forecast) - len(forecast))
            y_pred = forecast.tail(len(forecast) - len(forecast))
            total_mse += mean_squared_error(y_true[col], y_pred['yhat'])
            total_mae += mean_absolute_error(y_true[col], y_pred['yhat'])

        results = {
            "model": prophet_models,
            "mse": total_mse / len(numerical_cols),
            "mae": total_mae / len(numerical_cols),
            "forecasts": prophet_forecasts,
        }

    else:
        raise ValueError("Invalid model type")

    return results

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

        model_type = st.selectbox(
            "Select the model", ["LSTM", "XGBoost", "Prophet"]
        )
        seq_length = (
            st.slider("LSTM Sequence Length", min_value=1, max_value=30, value=10)
            if model_type == "LSTM"
            else None
        )

        if st.button("Train and Predict"):
            try:
                results = train_and_evaluate(df, model_type, seq_length)

                st.subheader("Evaluation Metrics")
                st.write(f"Mean Squared Error: {results['mse']:.4f}")
                st.write(f"Mean Absolute Error: {results['mae']:.4f}")

                st.subheader("Predictions/Forecasts")
                if model_type != "Prophet":
                    results_df = pd.DataFrame(
                        results["y_test"], columns=[f"Actual_{col}" for col in df.select_dtypes(include=[np.number]).columns.tolist() if col != 'Date']
                    )
                    pred_df = pd.DataFrame(
                        results["y_pred"], columns=[f"Predicted_{col}" for col in df.select_dtypes(include=[np.number]).columns.tolist() if col != 'Date']
                    )
                    final_df = pd.concat([results_df, pred_df], axis=1)
                    st.dataframe(final_df)
                    st.line_chart(final_df)  # Visualize all predictions

                    # Save model and scaler
                    joblib.dump(results["model"], f'trained_{model_type}_model.joblib')
                    if model_type != "Prophet":
                        joblib.dump(results["scaler"], f'scaler.joblib')
                    st.success(f"Trained {model_type} model and scaler saved.")
                else:
                    for col, forecast in results["forecasts"].items():
                        st.write(f"Forecast for {col}:")
                        forecast_df = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30)  # Show last 30 days
                        st.dataframe(forecast_df)
                        st.line_chart(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].set_index('ds'))

            except Exception as e:
                st.error(f"Error during training/prediction: {e}")

    except Exception as e:
        st.error(f"Error processing file: {e}")