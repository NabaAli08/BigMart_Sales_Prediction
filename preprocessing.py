import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def preprocess_data(df):

    df = df.copy()

    # Handle missing values safely (only if column exists)
    if "Item_Weight" in df.columns:
        df["Item_Weight"] = df["Item_Weight"].fillna(
            df["Item_Weight"].median()
        )

    if "Item_Visibility" in df.columns:
        df["Item_Visibility"] = df["Item_Visibility"].replace(
            0, df["Item_Visibility"].mean()
        )

    if "Outlet_Size" in df.columns:
        df["Outlet_Size"] = df["Outlet_Size"].fillna(
            df["Outlet_Size"].mode()[0]
        )

    # Feature Engineering (only if year exists)
    if "Outlet_Establishment_Year" in df.columns:
        df["Item_Age"] = 2013 - df["Outlet_Establishment_Year"]

    # Drop unnecessary columns safely
    columns_to_drop = [
        "Item_Identifier",
        "Outlet_Identifier",
        "Outlet_Establishment_Year",
    ]

    df = df.drop(
        columns=[col for col in columns_to_drop if col in df.columns]
    )

    # Encode categorical variables
    cat_cols = df.select_dtypes(include=["object"]).columns

    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])

    return df
