from flask import Flask, render_template, request
import pandas as pd
import joblib
from src.preprocessing import preprocess_data
import numpy
import json

app = Flask(__name__)

model = joblib.load("models/best_model.pkl")


@app.route("/")
def home():
    return render_template("index.html")


import numpy as np
import json

@app.route("/predict", methods=["POST"])
def predict():

    input_data = {
        "Item_Weight": float(request.form["Item_Weight"]),
        "Item_Fat_Content": request.form["Item_Fat_Content"],
        "Item_Visibility": float(request.form["Item_Visibility"]),
        "Item_Type": request.form["Item_Type"],
        "Item_MRP": float(request.form["Item_MRP"]),
        "Outlet_Establishment_Year": int(request.form["Outlet_Establishment_Year"]),
        "Outlet_Size": request.form["Outlet_Size"],
        "Outlet_Location_Type": request.form["Outlet_Location_Type"],
        "Outlet_Type": request.form["Outlet_Type"],
    }

    df = pd.DataFrame([input_data])
    df = preprocess_data(df)

    prediction = model.predict(df)[0]

    feature_names = df.columns.tolist()
    feature_importance = model.feature_importances_.tolist()

    return render_template(
        "index.html",
        prediction=round(prediction, 2),
        feature_names=json.dumps(feature_names),
        feature_importance=json.dumps(feature_importance),
    )

if __name__ == "__main__":
    app.run(debug=True)
