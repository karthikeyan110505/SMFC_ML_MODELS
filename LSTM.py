# %% [markdown]
# Import libraries and modules

# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score

# %% [markdown]
# Load the dataset

# %%

data = {
    'TIME': ['day-1; 1pm', 'day-1; 7pm', 'day-2; 1am', 'day-2; 7am', 'day-2; 1pm', 'day-2; 7pm', 'day-3; 1am', 'day-3; 7am', 'day-3; 1pm', 'day-3; 7pm',
             'day-4; 1am', 'day-4; 7am', 'day-4; 1pm', 'day-4; 7pm', 'day-5; 1am', 'day-5; 7am', 'day-5; 1pm', 'day-5; 7pm', 'day-6; 1am', 'day-6; 7am',
             'day-6; 1pm', 'day-6; 7pm', 'day-7; 1am', 'day-7; 7am', 'day-7; 1pm', 'day-7; 7pm', 'day-8; 1am', 'day-8; 7am', 'day-8; 1pm', 'day-8; 7pm',
             'day-9; 1am', 'day-9; 7am', 'day-9; 1pm', 'day-9; 7pm', 'day-10; 1am', 'day-10; 7am', 'day-10; 1pm', 'day-10; 7pm', 'day-11; 1am', 'day-11; 7am',
             'day-11; 1pm', 'day-11; 7pm', 'day-12; 1am', 'day-12; 7am', 'day-12; 1pm', 'day-12; 7pm', 'day-13; 1am', 'day-13; 7am', 'day-13; 1pm', 'day-13; 7pm',
             'day-14; 1am', 'day-14; 7am', 'day-14; 1pm', 'day-14; 7pm', 'day-15; 1am', 'day-15; 7am', 'day-15; 1pm', 'day-15; 7pm', 'day-16; 1am', 'day-16; 7am',
             'day-16; 1pm', 'day-16; 7pm', 'day-17; 1am', 'day-17; 7am', 'day-17; 1pm', 'day-17; 7pm', 'day-18; 1am', 'day-18; 7am', 'day-18; 1pm', 'day-18; 7pm',
             'day-19; 1am', 'day-19; 7am', 'day-19; 1pm', 'day-19; 7pm', 'day-20; 1am', 'day-20; 7am', 'day-20; 1pm', 'day-20; 7pm', 'day-21; 1am', 'day-21; 7am',
             'day-21; 1pm', 'day-21; 7pm', 'day-22; 1am', 'day-22; 7am', 'day-22; 1pm', 'day-22; 7pm', 'day-23; 1am', 'day-23; 7am', 'day-23; 1pm', 'day-23; 7pm',
             'day-24; 1am', 'day-24; 7am', 'day-24; 1pm', 'day-24; 7pm', 'day-25; 1am', 'day-25; 7am', 'day-25; 1pm', 'day-25; 7pm', 'day-26; 1am', 'day-26; 7am',
             'day-26; 1pm', 'day-26; 7pm'],
    'VOLTAGE': [0.75, 0.706, 0.607, 0.456, 0.389, 0.349, 0.244, 0.187, 0.166, 0.159, 0.11, 0.098, 0.081, 0.083, 0.046, 0.055, 0.051, 0.484, 0.483, 0.62,
                0.511, 0.478, 0.48, 0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62, 0.64, 0.66, 0.68, 0.7, 0.72, 0.74, 0.76, 0.78, 0.456, 0.62, 0.654, 0.625, 0.625, 0.515,
                0.52, 0.625, 0.615, 0.628, 0.615, 0.628, 0.578, 0.634, 0.635, 0.645, 0.72, 0.8, 0.88, 0.946, 0.84, 0.894, 0.9, 0.89, 0.885, 0.88, 0.89, 0.9,
                0.905, 0.91, 0.99, 0.995, 0.91, 0.92, 0.921, 0.923, 0.92, 0.92, 0.921, 0.92, 0.915, 0.925, 0.918, 0.908, 0.915, 0.923, 0.925, 0.93, 0.928, 0.93,
                0.932, 0.935, 0.938, 0.94, 0.942, 0.956, 0.948, 0.95, 0.94, 0.91, 0.935, 0.946, 0.957, 0.961]
}

df = pd.DataFrame(data)


# %% [markdown]
# Proper datetime formatting

# %%

def parse_datetime(day_time_str):
    day, time = day_time_str.split('; ')
    day = int(day.replace('day-', ''))
    return pd.to_datetime(f'2025-03-{day:02d} {time}', format='%Y-%m-%d %I%p')

df['TIME'] = df['TIME'].apply(parse_datetime)
df = df.sort_values('TIME')

# %% [markdown]
# Normalize the voltage data

# %%

scaler = MinMaxScaler(feature_range=(0, 1))
df['VOLTAGE'] = scaler.fit_transform(df['VOLTAGE'].values.reshape(-1, 1))

# %% [markdown]
# Function to create sequences

# %%

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i + seq_length])
        ys.append(data[i + seq_length])
    return np.array(xs), np.array(ys)

# %% [markdown]
# Create sequences

# %%

SEQ_LENGTH = 10
X, y = create_sequences(df['VOLTAGE'].values, SEQ_LENGTH)

# Reshape for LSTM
X = np.expand_dims(X, axis=-1)

# %% [markdown]
# Split into training and test sets

# %%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [markdown]
# Build the LSTM model

# %%

model = Sequential([
    Input(shape=(SEQ_LENGTH, 1)),
    LSTM(64, return_sequences=True),
    LSTM(64),
    Dense(32, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')

# %% [markdown]
# Train the model

# %%

history = model.fit(X_train, y_train, batch_size=16, epochs=20, validation_data=(X_test, y_test), verbose=0)

# %% [markdown]
# Predict on test data

# %%

y_pred = model.predict(X_test)

# %% [markdown]
# Inverse transform predictions and true values

# %%

y_pred_inv = scaler.inverse_transform(y_pred)
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# %% [markdown]
# Calculate metrics

# %%

mse = mean_squared_error(y_test_inv, y_pred_inv)
r2 = r2_score(y_test_inv, y_pred_inv)

print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'R² Score: {r2:.4f}')

# %% [markdown]
# Plot predicted vs actual

# %%

plt.figure(figsize=(14, 6))
plt.plot(df['TIME'][-len(y_test_inv):], y_test_inv, label='True Voltage')
plt.plot(df['TIME'][-len(y_test_inv):], y_pred_inv, label='Predicted Voltage', linestyle='dashed')
plt.xlabel('Time')
plt.ylabel('Voltage')
plt.legend()
plt.title('Voltage Prediction with LSTM')
plt.show()

# %% [markdown]
# Predict future voltage values

# %%

future_steps = 10
last_sequence = df['VOLTAGE'].values[-SEQ_LENGTH:].reshape(1, SEQ_LENGTH, 1)

future_voltages = []
for _ in range(future_steps):
    next_voltage = model.predict(last_sequence, verbose=0)[0][0]
    future_voltages.append(next_voltage)
    last_sequence = np.roll(last_sequence, -1)
    last_sequence[0, -1, 0] = next_voltage

# %% [markdown]
# Inverse scale

# %%

future_voltages = scaler.inverse_transform(np.array(future_voltages).reshape(-1, 1))

# %% [markdown]
# Future time steps

# %%

future_dates = pd.date_range(df['TIME'].iloc[-1] + pd.Timedelta(hours=7), periods=future_steps, freq='7h')

# %% [markdown]
# Plot future prediction

# %%

plt.figure(figsize=(14, 6))
plt.plot(df['TIME'], scaler.inverse_transform(df['VOLTAGE'].values.reshape(-1, 1)), label='Historical Voltage')
plt.plot(future_dates, future_voltages, label='Future Predicted Voltage', linestyle='dashed', color='orange')
plt.xlabel('Time')
plt.ylabel('Voltage')
plt.legend()
plt.title('Future Voltage Prediction')
plt.show()

# %% [markdown]
# Print future predictions

# %%

print("\nFuture Voltage Predictions:")
for date, voltage in zip(future_dates, future_voltages):
    print(f"{date.strftime('%Y-%m-%d %I:%M %p')}: {voltage[0]:.4f} V")

# %% [markdown]
# Evaluate on train/test

# %%

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# %% [markdown]
# Inverse transform

# %%

y_train_pred = scaler.inverse_transform(y_train_pred)
y_train = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_pred = scaler.inverse_transform(y_test_pred)
y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

# %% [markdown]
# R² scores

# %%

r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

print(f"Train R² Score: {r2_train:.4f}")
print(f"Test R² Score: {r2_test:.4f}")

# %% [markdown]
# Plot training and testing predictions

# %%

fig, axs = plt.subplots(2, 1, figsize=(14, 10))

axs[0].plot(y_train, label='True Voltage (Train)', color='blue')
axs[0].plot(y_train_pred, label='Predicted Voltage (Train)', linestyle='dashed', color='green')
axs[0].set_title(f'Training Set - R²: {r2_train:.4f}')
axs[0].set_xlabel('Sample')
axs[0].set_ylabel('Voltage (V)')
axs[0].legend()

axs[1].plot(y_test, label='True Voltage (Test)', color='blue')
axs[1].plot(y_test_pred, label='Predicted Voltage (Test)', linestyle='dashed', color='red')
axs[1].set_title(f'Testing Set - R²: {r2_test:.4f}')
axs[1].set_xlabel('Sample')
axs[1].set_ylabel('Voltage (V)')
axs[1].legend()

plt.tight_layout()
plt.show()



