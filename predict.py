import pandas as pd
import joblib
from preprocessing import preprocess_data

# Load model
model = joblib.load("models/best_model.pkl")

# Load new data
new_data = pd.read_csv("data/train.csv")

new_data = preprocess_data(new_data, training=False)

X_new = new_data.drop("Item_Outlet_Sales", axis=1)

predictions = model.predict(X_new)

print("Sample Predictions:")
print(predictions[:10])
