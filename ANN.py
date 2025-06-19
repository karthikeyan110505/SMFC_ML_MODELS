# %% [markdown]
# Import Libraries

# %%
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

# %% [markdown]
# Reload the dataset

# %%

data = {
    'TIME': ['day-1; 1pm', 'day-1; 7pm', 'day-2; 1am', 'day-2; 7am', 'day-2; 1pm', 'day-2; 7pm',
             'day-3; 1am', 'day-3; 7am', 'day-3; 1pm', 'day-3; 7pm'] * 10,  # Simulating 100 samples
    'VOLTAGE': [0.75, 0.706, 0.607, 0.456, 0.389, 0.349, 0.244, 0.187, 0.166, 0.159,
                0.11, 0.098, 0.081, 0.083, 0.046, 0.055, 0.051, 0.484, 0.483, 0.62,
                0.511, 0.478, 0.48, 0.5, 0.52, 0.54, 0.56, 0.58, 0.6, 0.62,
                0.64, 0.66, 0.68, 0.7, 0.72, 0.74, 0.76, 0.78, 0.456, 0.62,
                0.654, 0.625, 0.625, 0.515, 0.52, 0.625, 0.615, 0.628, 0.615, 0.628,
                0.578, 0.634, 0.635, 0.645, 0.72, 0.8, 0.88, 0.946, 0.84, 0.894,
                0.9, 0.89, 0.885, 0.88, 0.89, 0.9, 0.905, 0.91, 0.99, 0.995,
                0.91, 0.92, 0.921, 0.923, 0.92, 0.92, 0.921, 0.92, 0.915, 0.925,
                0.918, 0.908, 0.915, 0.923, 0.925, 0.93, 0.928, 0.93, 0.932, 0.935,
                0.938, 0.94, 0.942, 0.956, 0.948, 0.95, 0.94, 0.91, 0.935, 0.946]
}

# Create the DataFrame
df = pd.DataFrame(data)

# %% [markdown]
# Convert TIME column to datetime

# %%

def parse_datetime(day_time_str):
    day, time = day_time_str.split('; ')
    day = int(day.replace('day-', ''))
    return pd.to_datetime(f'2025-03-{day} {time}', format='%Y-%m-%d %I%p')

df['TIME'] = df['TIME'].apply(parse_datetime)

# %% [markdown]
# Sort the dataframe by time

# %%

df = df.sort_values('TIME')

# Feature Engineering
df['TIME_SIN'] = np.sin(2 * np.pi * df['TIME'].dt.hour / 24)
df['TIME_COS'] = np.cos(2 * np.pi * df['TIME'].dt.hour / 24)

# %% [markdown]
# Normalize the voltage data

# %%

scaler = MinMaxScaler(feature_range=(0, 1))
df['VOLTAGE'] = scaler.fit_transform(df['VOLTAGE'].values.reshape(-1, 1))

# %% [markdown]
# Prepare features and target

# %%

X = df[['TIME_SIN', 'TIME_COS']].values
y = df['VOLTAGE'].values

# %% [markdown]
# Split into training and test sets

# %%

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [markdown]
# Train the ANN model using MLPRegressor

# %%

mlp_model = MLPRegressor(hidden_layer_sizes=(64, 128, 64), activation='relu', max_iter=150, random_state=42)
mlp_model.fit(X_train, y_train)

# %% [markdown]
# Predict on the test set

# %%

y_pred = mlp_model.predict(X_test)

# %% [markdown]
# Inverse transform the predictions and actual values

# %%

y_pred_inv = scaler.inverse_transform(y_pred.reshape(-1, 1))
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# %% [markdown]
# Calculate MSE and R² score

# %%

mse = mean_squared_error(y_test_inv, y_pred_inv)
r2 = r2_score(y_test_inv, y_pred_inv)

# Print the results
print(f'Mean Squared Error (MSE): {mse:.4f}')
print(f'R² Score: {r2:.4f}')

# %% [markdown]
# Plot actual vs predicted

# %%

plt.figure(figsize=(14, 6))
plt.plot(y_test_inv, label='True Voltage', color='blue')
plt.plot(y_pred_inv, label='Predicted Voltage', linestyle='dashed', color='orange')
plt.xlabel('Samples')
plt.ylabel('Voltage (V)')
plt.legend()
plt.title('ANN Voltage Prediction using MLPRegressor')
plt.show()

# %% [markdown]
# Future prediction

# %%

future_steps = 10
future_hours = np.arange(1, future_steps + 1) * 7  # 7-hour intervals
future_sin = np.sin(2 * np.pi * future_hours / 24)
future_cos = np.cos(2 * np.pi * future_hours / 24)
future_X = np.column_stack((future_sin, future_cos))

# %% [markdown]
# Make future predictions

# %%

future_pred = mlp_model.predict(future_X)
future_pred_inv = scaler.inverse_transform(future_pred.reshape(-1, 1))

# %% [markdown]
# Create future dates for plotting and Plot future predictions

# %%

future_dates = pd.date_range(df['TIME'].iloc[-1] + pd.Timedelta(hours=7), periods=future_steps, freq='7h')

plt.figure(figsize=(14, 6))
plt.plot(df['TIME'], scaler.inverse_transform(df['VOLTAGE'].values.reshape(-1, 1)), label='Historical Voltage')
plt.plot(future_dates, future_pred_inv, label='Future Predicted Voltage', linestyle='dashed', color='green')
plt.xlabel('Time')
plt.ylabel('Voltage (V)')
plt.legend()
plt.title('Future Voltage Prediction with MLPRegressor')
plt.show()

# %% [markdown]
# Train and test predictions

# %%

y_train_pred = mlp_model.predict(X_train)
y_train_pred_inv = scaler.inverse_transform(y_train_pred.reshape(-1, 1))
y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))

# %% [markdown]
# Calculate R² scores

# %%

r2_train = r2_score(y_train_inv, y_train_pred_inv)
r2_test = r2_score(y_test_inv, y_pred_inv)

mse_1 = mean_squared_error(y_train_inv, y_train_pred_inv)
mse_2 = mean_squared_error(y_test_inv, y_pred_inv)

# %% [markdown]
# Print future predictions

# %%

future_results = list(zip(future_dates, future_pred_inv.flatten()))
future_results[:10]  # Displaying the first 10 future predictions for verification

# %% [markdown]
# Plot Train Data, Plot Test Data and Print R² scores

# %%

plt.figure(figsize=(14, 6))
plt.plot(y_train_inv, label='True Voltage (Train)', color='blue')
plt.plot(y_train_pred_inv, label='Predicted Voltage (Train)', linestyle='dashed', color='orange')
plt.xlabel('Samples')
plt.ylabel('Voltage (V)')
plt.legend()
plt.title(f'ANN Voltage Prediction (Train) - R²: {r2_train:.4f}')
plt.show()

plt.figure(figsize=(14, 6))
plt.plot(y_test_inv, label='True Voltage (Test)', color='green')
plt.plot(y_pred_inv, label='Predicted Voltage (Test)', linestyle='dashed', color='red')
plt.xlabel('Samples')
plt.ylabel('Voltage (V)')
plt.legend()
plt.title(f'ANN Voltage Prediction (Test) - R²: {r2_test:.4f}')
plt.show()

print(f'R² Score (Train): {r2_train:.4f}')
print(f'R² Score (Test): {r2_test:.4f}')
print(f'Mean Squared Error (train): {mse_1:.4f}')
print(f'Mean Squared Error (test): {mse_2:.4f}')


