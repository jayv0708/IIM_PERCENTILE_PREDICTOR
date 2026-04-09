import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

print("Loading IIM Admission Data...")
data = pd.read_csv("iim_admission_predict.csv")

# Identify target columns
target_cols = [c for c in data.columns if c.startswith('Target_')]
X = data.drop(target_cols, axis=1)
y = data[target_cols]

# One-hot encode the Categorical features
X = pd.get_dummies(X, columns=['Category', 'Gender', 'Undergrad_Stream'], drop_first=False)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

print(f"Training Multi-Output Random Forest Regressor on {len(target_cols)} targets...")
model = RandomForestRegressor(
    n_estimators=100,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Abs. Error across all IIMs: {mae:.2f} percentiles")

# Save model, feature order, and target list
joblib.dump(model, "model.pkl")
joblib.dump(list(X.columns), "features.pkl")
joblib.dump(target_cols, "targets.pkl")
print("Model, features, and targets saved successfully!")