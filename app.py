import yfinance as yf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import datetime

def get_stock_data(symbol, start_date, end_date):
    stock_data = yf.download(symbol, start=start_date, end=end_date)
    return stock_data['Close'].values.reshape(-1, 1)

def create_features_and_labels(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i+window_size])
        y.append(data[i+window_size])
    return np.array(X), np.array(y)

def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    return mse

def predict_future_price(model, last_window, prediction_steps):
    forecast = []
    latest_window = last_window.reshape(1, -1)
    for i in range(prediction_steps):
        next_prediction = model.predict(latest_window)
        forecast.append(next_prediction[0])
        latest_window = np.append(latest_window[:, 1:], next_prediction.reshape(1, -1), axis=1)
    return forecast

if __name__ == "__main__":
    symbol = 'AAPL'
    start_date = datetime.datetime(2020, 1, 1)
    end_date = datetime.datetime.now()
    window_size = 10
    prediction_steps = 5

    data = get_stock_data(symbol, start_date, end_date)
    X, y = create_features_and_labels(data, window_size)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = train_model(X_train, y_train)
    mse = evaluate_model(model, X_test, y_test)
    print("Mean Squared Error:", mse)

    last_window = data[-window_size:]
    forecast = predict_future_price(model, last_window, prediction_steps)
    print("Predicted prices for the next {} steps: {}".format(prediction_steps, forecast))
