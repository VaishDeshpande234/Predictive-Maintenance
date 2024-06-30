# Predictive Maintenance for Turbofan Engine using Machine Learning

# Project Overview

The goal of this project is to predict the Remaining Useful Life (RUL) of Turbofan engines based on sensor data. Predictive maintenance helps identify the point at which an engine is likely to fail, allowing for timely maintenance to prevent failures and optimize maintenance schedules.

# Dataset
The dataset used in this project is the FD001 dataset from the NASA Prognostics Data Repository. It contains sensor data collected from multiple aircraft engines. Each engine has a different number of cycles of operation until failure.

# Dataset Files

train_FD001.txt: Training dataset with sensor readings up to the failure point.
test_FD001.txt: Test dataset with sensor readings up to a certain point, without RUL values.
RUL_FD001.txt: Ground truth RUL values for the engines in the test dataset.

# Features

The datasets contain the following features:

unit_number
time_in_cycles
setting_1
setting_2
TRA
T2
T24
T30
T50
P2
P15
P30
Nf
Nc
epr
Ps30
phi
NRf
NRc
BPR
farB
htBleed
Nf_dmd
PCNfR_dmd
W31
W32

# Project Structure

Data Preprocessing and Exploratory Data Analysis (EDA)

Data Loading: Load the training, test, and RUL datasets.
- Data Cleaning: Drop unnecessary columns and rename columns for clarity.
- EDA: Analyze the distributions of sensor readings, identify correlations, and visualize the data.

Feature Engineering
- Calculate RUL: For the training dataset, calculate the RUL based on the maximum cycle time for each unit.
- Prepare Test Dataset: Merge the RUL values with the test dataset.

Model Training and Evaluation
- Train/Test Split: Split the data into features and target variables.
- Model Training: Train Random Forest, Gradient Boosting, and LSTM models.
- Model Evaluation: Evaluate the models using RMSE and MAE metrics.

# Results

The models are evaluated based on their RMSE and MAE scores:

Random Forest: RMSE: 46.35, MAE: 35.01
Gradient Boosting: RMSE: 45.76, MAE: 34.44
LSTM: RMSE: 47.04, MAE: 35.50

# Visualisation

## 1. Correlation Matrix of Features

### Description:
The correlation matrix visualizes the Pearson correlation coefficients between pairs of features in the dataset. Positive values indicate a direct relationship, while negative values indicate an inverse relationship. Values close to 1 or -1 indicate strong correlations.

### Key Highlights:

#### Strong positive correlations between:
- T24 and T50 (0.71)
- T30 and T50 (0.68)
- P30 and Ps30 (0.82)
- Nf and Nc (0.83)

#### Strong negative correlations between:
- P30 and phi (-0.82)
- Ps30 and phi (-0.85)
- Nf and phi (-0.79)
- Nc and phi (-0.79)

## 2. Distribution of Time in Cycles (Train)

### Description:
This histogram shows the distribution of engine cycles in the training dataset. The x-axis represents the number of cycles, and the y-axis represents the frequency of engines operating for a given number of cycles.

### Key Highlights:

- Higher frequency of engines with cycle counts between 0 to around 170 cycles.
- Gradual decline in the number of engines beyond 170 cycles, with very few engines reaching up to 350 cycles.
- Indicates most engines tend to fail or are serviced before very high cycle counts.

## 3. Predicted RUL vs. Ground Truth RUL

### Description:
This scatter plot compares the predicted Remaining Useful Life (RUL) against the ground truth RUL for three different models: Random Forest, Gradient Boosting, and LSTM. Each point represents a prediction for a specific engine cycle, with the dashed line indicating a perfect prediction (i.e., predicted RUL equals ground truth RUL).

### Key Highlights:

- Random Forest (blue points): Predictions cluster around the ground truth with some variance.
- Gradient Boosting (orange points): Predictions are close to the ground truth with some spread.
- LSTM (green points): Densely packed predictions with more variance and some outliers.
