import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import pickle

# Load the dataset
df = pd.read_csv('car_data_set.csv')  # Make sure this is the correct file name

# Encode categorical features
le_brand = LabelEncoder()
le_fuel = LabelEncoder()

df['Brand'] = le_brand.fit_transform(df['Brand'])
df['Fuel'] = le_fuel.fit_transform(df['Fuel'])

# Create car age feature
df['Age'] = 2025 - df['Year']
df = df.drop('Year', axis=1)

# Define X and y
X = df[['Brand', 'Age', 'KM_Driven', 'Fuel']]
y = df['Selling_Price']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------- Linear Regression ----------
model_lr = LinearRegression()
model_lr.fit(X_train, y_train)

# Evaluate
y_pred_lr = model_lr.predict(X_test)
print("Linear Regression R²:", r2_score(y_test, y_pred_lr))
print("Linear Regression MSE:", mean_squared_error(y_test, y_pred_lr))

# Save linear model
with open('model_lr.pkl', 'wb') as f:
    pickle.dump(model_lr, f)

# ---------- Random Forest ----------
model_rf = RandomForestRegressor(n_estimators=100, random_state=42)
model_rf.fit(X_train, y_train)

# Evaluate
y_pred_rf = model_rf.predict(X_test)
print("Random Forest R²:", r2_score(y_test, y_pred_rf))
print("Random Forest MSE:", mean_squared_error(y_test, y_pred_rf))

# Save RF model
with open('model_rf.pkl', 'wb') as f:
    pickle.dump(model_rf, f)

# ---------- Save encoders ----------
with open('brand_encoder.pkl', 'wb') as f:
    pickle.dump(le_brand, f)

with open('fuel_encoder.pkl', 'wb') as f:
    pickle.dump(le_fuel, f)

print("✅ Models and encoders saved successfully.")
