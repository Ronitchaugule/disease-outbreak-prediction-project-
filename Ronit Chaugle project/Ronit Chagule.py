import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
df = pd.read_csv(url)

# Streamlit UI
st.title("COVID-19 Outbreak Prediction Dashboard")
st.sidebar.header("Settings")

# Select country
country = st.sidebar.selectbox("Select a Country:", df["Country/Region"].unique())

# Process data for the selected country
df = df[df["Country/Region"] == country].iloc[:, 4:].T
df.columns = ["Cases"]
df.index = pd.to_datetime(df.index)
df = df.diff().dropna()  # Convert to daily cases

# Plot actual cases
st.subheader(f"Daily COVID-19 Cases in {country}")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df, label="Actual Cases")
ax.set_title(f"COVID-19 Daily Cases - {country}")
ax.legend()
st.pyplot(fig)

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
df_scaled = scaler.fit_transform(df)

# Create sequences
def create_sequences(data, look_back=14):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i : i + look_back])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)

look_back = 14
X, y = create_sequences(df_scaled, look_back)

# Reshape for LSTM
X = X.reshape((X.shape[0], X.shape[1], 1))

# Split into train-test sets
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Define & Train LSTM Model
model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(look_back, 1)),
    LSTM(50, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X_train, y_train, epochs=10, batch_size=8, verbose=0)

# Predictions
y_pred = model.predict(X_test)
y_pred = scaler.inverse_transform(y_pred)
y_test = scaler.inverse_transform(y_test)

# Display Prediction Graph
st.subheader(f"Predicted vs Actual Cases for {country}")
fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.plot(y_test, label="Actual Cases")
ax2.plot(y_pred, label="Predicted Cases", linestyle="dashed")
ax2.legend()
ax2.set_title("COVID-19 Prediction using LSTM")
st.pyplot(fig2)

# Future Forecast (Next 7 Days)
future_preds = []
last_14_days = df_scaled[-look_back:]

for _ in range(7):
    input_data = last_14_days.reshape(1, look_back, 1)
    pred = model.predict(input_data)
    future_preds.append(pred[0][0])
    last_14_days = np.append(last_14_days[1:], pred, axis=0)

future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

# Display Future Prediction
st.subheader("Future Prediction (Next 7 Days)")
st.write(pd.DataFrame(future_preds, columns=["Predicted Cases"]).set_index(pd.date_range(start=df.index[-1], periods=7, freq='D')))

#cd "C:\Users\Ronit Chaugule\PycharmProjects\pythonProject5"
# streamlit run covid_dashboard.py