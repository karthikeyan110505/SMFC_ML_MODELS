# %% [markdown]
# Import libraries and modules

# %%
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# %% [markdown]
# Load the data from CSV

# %%
data = pd.read_csv('data_smfc_data.csv')

# %% [markdown]
# Convert TIME to datetime and extract features

# %%
data['TIME'] = pd.to_datetime(data['TIME'], format='day-%d; %I%p')
data['Day'] = data['TIME'].dt.day
data['Hour'] = data['TIME'].dt.hour

# %% [markdown]
# Create lag features

# %%
data['VOLTAGE_LAG_1'] = data['VOLTAGE'].shift(1)
data['VOLTAGE_LAG_2'] = data['VOLTAGE'].shift(2)
data = data.dropna()

# %% [markdown]
# Prepare the data

# %%
X = data[['Day', 'Hour', 'VOLTAGE_LAG_1', 'VOLTAGE_LAG_2']]
y = data['VOLTAGE']

# %% [markdown]
# Split the data

# %%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %% [markdown]
# Train the model

# %%
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# %% [markdown]
# Test the model

# %%
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# %% [markdown]
# Predict future voltages

# %%
future_days = np.arange(27, 33)  # Predict for days 27 to 32
future_hours = [1, 7, 13, 19]  # Same hours as in the dataset

# %% [markdown]
# Initialize with the last known data point

# %%
last_known_voltage = data.iloc[-1]['VOLTAGE']
last_known_voltage_lag_1 = data.iloc[-1]['VOLTAGE_LAG_1']
last_known_voltage_lag_2 = data.iloc[-1]['VOLTAGE_LAG_2']

future_predictions = []
for day in future_days:
    for hour in future_hours:
        future_data = pd.DataFrame([(day, hour, last_known_voltage_lag_1, last_known_voltage_lag_2)], columns=['Day', 'Hour', 'VOLTAGE_LAG_1', 'VOLTAGE_LAG_2'])
        future_voltage = model.predict(future_data)[0]
        future_predictions.append([day, hour,future_voltage])
        last_known_voltage_lag_2 = last_known_voltage_lag_1
        last_known_voltage_lag_1 = last_known_voltage
        last_known_voltage = future_voltage

# %% [markdown]
# Display future predictions

# %%
for day, hour, voltage in future_predictions:
    print(f'Day {day}, Hour {hour}: Predicted Voltage = {voltage:.3f} V')

# %% [markdown]
# Predict on training set for plotting

# %%
y_train_pred = model.predict(X_train)
y_test_pred = y_pred  # Already predicted earlier

# %% [markdown]
# R² for train/test sets

# %%
r2_train = r2_score(y_train, y_train_pred)
r2_test = r2_score(y_test, y_test_pred)

# %% [markdown]
# Calculate R² scores

# %%
print(" ")
print(f'Training R² Score: {r2_train:.4f}')
print(f'Test R² Score: {r2_test:.4f}')

# %% [markdown]
# Plotting the results of Training set plot and Test set

# %%
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
axs[0].scatter(y_train, y_train_pred, color='royalblue', alpha=0.6, label=f'R²: {r2_train:.4f}')
axs[0].plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', lw=2)  # y = x line
axs[0].set_title('Training Set')
axs[0].set_xlabel('Actual Voltage')
axs[0].set_ylabel('Predicted Voltage')
axs[0].legend()
    
axs[1].scatter(y_test, y_test_pred, color='seagreen', alpha=0.6, label=f'R²: {r2_test:.4f}')
axs[1].plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)  # y = x line
axs[1].set_title('Test Set')
axs[1].set_xlabel('Actual Voltage')
axs[1].set_ylabel('Predicted Voltage')
axs[1].legend()
    
plt.suptitle('Random Forest Regressor: Actual vs. Predicted Voltage')
plt.tight_layout()
plt.show()

# %% [markdown]
# Reshape future_prediction for heatmap

# %%
future_predictions_df = pd.DataFrame(future_predictions, columns=['Day', 'Hour', 'Voltage'])
heatmap_data = future_predictions_df.pivot(index='Day', columns='Hour', values='Voltage')

# %% [markdown]
# Create the heatmap

# %%
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis')
plt.title('Predicted voltage Heatmap')
plt.xlabel('Hour')
plt.ylabel('Day')
plt.show()

# %%



