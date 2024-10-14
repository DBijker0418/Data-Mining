# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 17:01:14 2024

@author: dtbij
"""

import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Step 1: Download Data using yfinance
stock_data = yf.download('AAPL', start='2015-01-01', end='2023-01-01')
closing_prices = stock_data[['Close']]

# Step 2: Scale Data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(closing_prices)

# Step 3: Create Training and Test Data
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]

def create_dataset(dataset, time_step=60):
    X, y = [], []
    for i in range(len(dataset) - time_step - 1):
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_dataset(train_data)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

# Step 4: Build and Train the LSTM Model
model = Sequential()
model.add(LSTM(units=100, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))  # Adding dropout layer to reduce overfitting
model.add(LSTM(units=100, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, batch_size=32, epochs=100)

# Step 5: Evaluate the LSTM Model on the Test Set
X_test, y_test = create_dataset(test_data)  # Create test dataset similar to training
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

test_predictions = model.predict(X_test)
test_predictions = scaler.inverse_transform(test_predictions)

# Calculate performance metrics for LSTM
print("LSTM Model Evaluation Metrics:")
print("MSE:", mean_squared_error(y_test, test_predictions))
print("MAE:", mean_absolute_error(y_test, test_predictions))

# Step 6: Prepare data for Prophet
prophet_data = closing_prices.reset_index()[['Date', 'Close']]
prophet_data.columns = ['ds', 'y']

# Initialize and train the Prophet model with monthly seasonality
prophet_model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
prophet_model.add_seasonality(name='monthly', period=30.5, fourier_order=5)  # Adding monthly seasonality
prophet_model.fit(prophet_data)

# Function to get LSTM and Prophet predicted price for a specific date
def predict_price(input_date):
    # Convert input date to pandas datetime
    input_date = pd.to_datetime(input_date)

    # Prepare data for LSTM prediction
    last_data = scaled_data[-60:]  # Use the last 60 days of data for LSTM
    last_data = last_data.reshape(1, last_data.shape[0], 1)  # Reshape for LSTM
    lstm_prediction = model.predict(last_data)
    lstm_prediction = scaler.inverse_transform(lstm_prediction)[0][0]  # Inverse scale

    # Prepare data for Prophet prediction
    future = prophet_model.make_future_dataframe(periods=365 * 5)  # Extend for 5 years
    forecast = prophet_model.predict(future)

    # Check if input_date is within forecast range
    if input_date not in forecast['ds'].values:
        return "Date out of range. Please choose a date within the 5-year forecast range."

    # Get the predicted price for the input date from Prophet
    prophet_prediction = forecast[forecast['ds'] == input_date]['yhat'].values[0]

    return {
        "LSTM Prediction": lstm_prediction,
        "Prophet Prediction": prophet_prediction,
        "Input Date": input_date.date()
    }

# Function to return hybrid model prediction (average of LSTM and Prophet)
def hybrid_prediction(input_date):
    predictions = predict_price(input_date)
    lstm_pred = predictions['LSTM Prediction']
    prophet_pred = predictions['Prophet Prediction']
    hybrid_pred = (lstm_pred + prophet_pred) / 2  # Simple average of LSTM and Prophet predictions
    return {
        "LSTM Prediction": lstm_pred,
        "Prophet Prediction": prophet_pred,
        "Hybrid Prediction": hybrid_pred,
        "Input Date": predictions['Input Date']
    }

# Prompt for user input
input_date = input("Enter the date for prediction (YYYY-MM-DD): ")  # Ask for a date

# Predict prices using both LSTM and Prophet, and print them separately
predicted_prices = predict_price(input_date)
print(f"LSTM Prediction for {predicted_prices['Input Date']}: {predicted_prices['LSTM Prediction']}")
print(f"Prophet Prediction for {predicted_prices['Input Date']}: {predicted_prices['Prophet Prediction']}")

# Print the hybrid prediction by averaging LSTM and Prophet predictions
hybrid_predicted_prices = hybrid_prediction(input_date)
print(f"Hybrid Model Prediction for {hybrid_predicted_prices['Input Date']}: {hybrid_predicted_prices['Hybrid Prediction']}")
