import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

from preprocessing import preprocess_data

# Load Dataset
df = pd.read_csv("data/train.csv")

# Preprocess
df = preprocess_data(df)

# Separate features & target
X = df.drop("Item_Outlet_Sales", axis=1)
y = df["Item_Outlet_Sales"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Models
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Decision Tree": DecisionTreeRegressor(max_depth=6),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42)
}

best_model = None
best_score = -np.inf

print("Model Performance:\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    print(f"{name}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R2 Score: {r2:.4f}")
    print("-" * 30)

    if r2 > best_score:
        best_score = r2
        best_model = model

# Save best model
joblib.dump(best_model, "models/best_model.pkl")

print("Best model saved successfully!")
