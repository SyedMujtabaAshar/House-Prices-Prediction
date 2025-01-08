import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Step 1: Load Dataset
data_url = "C:/Users/Ashar Master/Desktop/Projects/House Pricing Prediction/HousePricing.csv"
data = pd.read_csv(data_url)

# Display basic info
print("Dataset Overview:\n", data.head())
print("\nDataset Info:\n")
data.info()

# Step 2: Data Preprocessing
# Checking for missing values
print("\nMissing Values:\n", data.isnull().sum())

# Fill missing values (if any) with median for simplicity
data = data.fillna(data.select_dtypes(include=[np.number]).median())

# Encoding categorical data (if any categorical columns exist)
categorical_cols = data.select_dtypes(include=['object']).columns
print("\nCategorical Columns:\n", categorical_cols)
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Converting Date to numerical value if it exists
if 'Date' in data.columns:
    data['Date'] = pd.to_datetime(data['Date']).map(pd.Timestamp.toordinal)

# Splitting features and target
y = data['SalePrice']
X = data.drop(columns=['SalePrice'])

# Step 3: Split Dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Build and Train Model
# Linear Regression
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Step 5: Evaluate Models
# Linear Regression Evaluation
y_pred_linear = linear_model.predict(X_test)
print("\nLinear Regression Metrics:")
print("MAE:", mean_absolute_error(y_test, y_pred_linear))
print("MSE:", mean_squared_error(y_test, y_pred_linear))
print("R2 Score:", r2_score(y_test, y_pred_linear))

# Random Forest Evaluation
y_pred_rf = rf_model.predict(X_test)
print("\nRandom Forest Metrics:")
print("MAE:", mean_absolute_error(y_test, y_pred_rf))
print("MSE:", mean_squared_error(y_test, y_pred_rf))
print("R2 Score:", r2_score(y_test, y_pred_rf))

# Step 6: Plot Results
plt.figure(figsize=(10, 5))
sns.scatterplot(x=y_test, y=y_pred_linear, label="Linear Regression")
sns.scatterplot(x=y_test, y=y_pred_rf, label="Random Forest")
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Ideal Fit')
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.legend()
plt.title("Comparison of Predicted vs Actual Prices")
plt.show()
