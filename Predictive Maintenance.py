#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


# In[3]:


#Load the dataset

train_df = pd.read_csv("C:/Users/welcome/Downloads/CMAPSSData/train_FD001.txt",sep=" ",header=None)
test_df = pd.read_csv("C:/Users/welcome/Downloads/CMAPSSData/test_FD001.txt",sep=" ",header=None)
rul_df = pd.read_csv("C:/Users/welcome/Downloads/CMAPSSData/RUL_FD001.txt",sep=" ",header=None)

train_df.describe()


# In[4]:


train_df.drop(columns=[26,27],inplace=True)
test_df.drop(columns=[26,27],inplace=True)


# In[5]:


columns = ['unit_number','time_in_cycles','setting_1','setting_2','TRA','T2','T24','T30','T50','P2','P15','P30','Nf',
           'Nc','epr','Ps30','phi','NRf','NRc','BPR','farB','htBleed','Nf_dmd','PCNfR_dmd','W31','W32' ]
train_df.columns = columns
test_df.columns = columns

# Rename RUL columns
rul_df.columns = ['RUL', 'NaN']
rul_df = rul_df.drop(columns=['NaN'])

# Display basic information about the datasets
print("Train Dataset Info:")
print(train_df.info())
print("\nTest Dataset Info:")
print(test_df.info())
print("\nRUL Dataset Info:")
print(rul_df.info())

# Display first few rows of each dataset
print("\nFirst 5 rows of Train Dataset:")
print(train_df.head())
print("\nFirst 5 rows of Test Dataset:")
print(test_df.head())
print("\nFirst 5 rows of RUL Dataset:")
print(rul_df.head())

# Summary statistics
print("\nTrain Dataset Statistics:")
print(train_df.describe())
print("\nTest Dataset Statistics:")
print(test_df.describe())


# In[6]:


# Plotting
# Time in cycles distribution
plt.figure(figsize=(12, 6))
sns.histplot(train_df['time_in_cycles'], bins=50, kde=True)
plt.title('Distribution of Time in Cycles (Train)')
plt.xlabel('Time in Cycles')
plt.ylabel('Frequency')
plt.show()


# In[7]:


# Settings and sensors distributions
columns_to_plot = ['setting_1', 'setting_2', 'TRA', 'T2', 'T24', 'T30', 'T50', 'P2', 'P15', 'P30', 'Nf', 'Nc', 'epr', 'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'farB', 'htBleed', 'Nf_dmd', 'PCNfR_dmd', 'W31', 'W32']
plt.figure(figsize=(16, 12))
for i, column in enumerate(columns_to_plot, 1):
    plt.subplot(6, 4, i)
    sns.histplot(train_df[column], bins=50, kde=True)
    plt.title(f'Distribution of {column}')
plt.tight_layout()
plt.show()


# In[8]:


#delete columns with constant values that do not carry information about the state of the unit
train_df.drop(columns=['Nf_dmd','PCNfR_dmd','P2','T2','TRA','farB','epr'],inplace=True)

#update columns to plot
columns_to_plot = ['setting_1', 'setting_2', 'T24', 'T30', 'T50', 'P15', 'P30', 'Nf', 'Nc', 'Ps30', 'phi', 'NRf', 'NRc', 'BPR', 'htBleed', 'W31', 'W32']

# Correlation matrix
plt.figure(figsize=(18, 12))
correlation_matrix = train_df[columns_to_plot].corr()
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm')
plt.title('Correlation Matrix of Features')
plt.show()


# In[9]:


# Function to calculate RUL for each unit in the training dataset
def calculate_rul(df):
    max_cycle = df.groupby('unit_number')['time_in_cycles'].max().reset_index()
    max_cycle.columns = ['unit_number', 'max_time_in_cycles']
    df = df.merge(max_cycle, on='unit_number', how='left')
    df['RUL'] = df['max_time_in_cycles'] - df['time_in_cycles']
    df = df.drop('max_time_in_cycles', axis=1)
    return df

# Calculate RUL for training dataset
train_df = calculate_rul(train_df)

# Display the first few rows to check the RUL calculation
print("\nFirst 5 rows of Train Dataset with RUL:")
print(train_df.head())

# Prepare the test dataset
# Test dataset's RUL values are in the RUL dataset
# Merge the RUL dataset with the test dataset

# Add unit_number column to RUL dataset
rul_df['unit_number'] = range(1, len(rul_df) + 1)

# Add RUL values to the test dataset
test_max_cycle = test_df.groupby('unit_number')['time_in_cycles'].max().reset_index()
test_max_cycle.columns = ['unit_number', 'max_time_in_cycles']
test_df = test_df.merge(test_max_cycle, on='unit_number', how='left')
test_df = test_df.merge(rul_df, on='unit_number', how='left')
test_df['RUL'] = test_df['RUL'] + test_df['max_time_in_cycles'] - test_df['time_in_cycles']
test_df = test_df.drop('max_time_in_cycles', axis=1)

# Display the first few rows to check the merge
print("\nFirst 5 rows of Test Dataset with RUL:")
print(test_df.head())

# Handle NaN values by filling with the median value
train_df.fillna(train_df.median(), inplace=True)
test_df.fillna(test_df.median(), inplace=True)


# In[11]:


# Select features and target
features = columns_to_plot
target = 'RUL'

X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]

# Train Random Forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

# Train Gradient Boosting model
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
gb_predictions = gb_model.predict(X_test)

# Evaluate models
def evaluate_model(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, mae

rf_rmse, rf_mae = evaluate_model(y_test, rf_predictions)
gb_rmse, gb_mae = evaluate_model(y_test, gb_predictions)

print(f"Random Forest - RMSE: {rf_rmse}, MAE: {rf_mae}")
print(f"Gradient Boosting - RMSE: {gb_rmse}, MAE: {gb_mae}")


# In[12]:


# Scale the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape for LSTM input [samples, time steps, features]
X_train_scaled = X_train_scaled.reshape(X_train_scaled.shape[0], 1, X_train_scaled.shape[1])
X_test_scaled = X_test_scaled.reshape(X_test_scaled.shape[0], 1, X_test_scaled.shape[1])

# Define LSTM model
lstm_model = Sequential()
lstm_model.add(LSTM(50, activation='relu', input_shape=(X_train_scaled.shape[1], X_train_scaled.shape[2])))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mse')

# Fit the model
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
lstm_model.fit(X_train_scaled, y_train, validation_split=0.2, epochs=100, batch_size=32, callbacks=[early_stopping], verbose=1)

# Make predictions
lstm_predictions = lstm_model.predict(X_test_scaled).flatten()

# Evaluate LSTM model
lstm_rmse, lstm_mae = evaluate_model(y_test, lstm_predictions)

print(f"LSTM - RMSE: {lstm_rmse}, MAE: {lstm_mae}")


# In[13]:


print("\nModel Performance Summary:")
print(f"Random Forest - RMSE: {rf_rmse:.2f}, MAE: {rf_mae:.2f}")
print(f"Gradient Boosting - RMSE: {gb_rmse:.2f}, MAE: {gb_mae:.2f}")
print(f"LSTM - RMSE: {lstm_rmse:.2f}, MAE: {lstm_mae:.2f}")


# In[14]:


# Extract ground truth RUL values from the test dataset
ground_truth = y_test

# Plotting
plt.figure(figsize=(12, 6))

# Plotting Random Forest predictions
plt.plot(ground_truth, rf_predictions, 'o', label='Random Forest', alpha=0.7)

# Plotting Gradient Boosting predictions
plt.plot(ground_truth, gb_predictions, 'o', label='Gradient Boosting', alpha=0.7)

# Plotting LSTM predictions
plt.plot(ground_truth, lstm_predictions, 'o', label='LSTM', alpha=0.7)

# Plotting the line of perfect prediction (y=x)
plt.plot([min(ground_truth), max(ground_truth)], [min(ground_truth), max(ground_truth)], 'k--', lw=2, label='Perfect Prediction')

plt.title('Predicted RUL vs. Ground Truth RUL')
plt.xlabel('Ground Truth RUL')
plt.ylabel('Predicted RUL')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

