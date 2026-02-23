# BigMart_Sales_Prediction
Predict BigMart product sales using machine learning! This project cleans and explores sales data, engineers features, and builds regression models to forecast sales, helping businesses make data-driven decisions and optimize inventory efficiently.



# BigMart Sales Prediction

A machine learning–based solution to predict item sales for BigMart stores using historical sales data. Helps store managers optimize inventory, improve supply chain efficiency, and make data-driven business decisions.

---

## Project Overview

This project leverages **Machine Learning (ML)** to forecast sales of products across multiple BigMart stores. By analyzing features such as item type, MRP, visibility, and store characteristics, the system builds predictive models that estimate future sales, enabling better inventory and business planning.

---

## ML Highlights

* **Regression Modeling:** Predicts item sales using Linear Regression, Decision Trees, Random Forest, and XGBoost.
* **Feature Engineering:** Encodes categorical variables, handles missing data, and derives new features such as store operation years.
* **Visualization:** Provides insights into sales patterns using plots and correlation analysis.
* **Evaluation:** Uses RMSE, MAE, and R² score to measure prediction performance.

---

## Repository Structure

```
BigMart_Sales_Prediction/
├── README.md               # Project overview and instructions
├── requirements.txt        # Python dependencies
├── data/
│   ├── train.csv           # Training dataset (with sales)
│   └── test.csv            # Test dataset (without sales)
├── notebooks/
│   ├── data_preprocessing.ipynb  # Cleaning, encoding, feature engineering
│   └── model_training.ipynb      # Model training, evaluation, predictions
├── models/
│   └── best_model.pkl       # Saved trained model
└── outputs/
    └── predictions.csv      # Model predictions on test data
```

---

## Dataset

Requires **BigMart sales dataset** (`train.csv` and `test.csv`). Place the CSV files inside a folder named `data/` at the root of the repo.

**Key Features:**

* `Item_Identifier`, `Item_Weight`, `Item_Fat_Content`, `Item_Visibility`
* `Item_Type`, `Item_MRP`, `Outlet_Identifier`
* `Outlet_Establishment_Year`, `Outlet_Size`, `Outlet_Location_Type`, `Outlet_Type`
* `Item_Outlet_Sales` (Target for training)

---

## Running the Project

1. Open Jupyter Notebook and run `data_preprocessing.ipynb` to:

   * Handle missing values
   * Encode categorical features
   * Create new features

2. Run `model_training.ipynb` to:

   * Split the data into training and validation sets
   * Train multiple models (Linear Regression, Decision Tree, Random Forest, XGBoost)
   * Evaluate models and save the best-performing model (`best_model.pkl`)

3. Generate predictions on test data:

   ```python
   # Example in notebook
   predictions = best_model.predict(test_data_preprocessed)
   predictions.to_csv("outputs/predictions.csv", index=False)
   ```

---

## Notes

* `best_model.pkl` → Serialized trained model for future predictions
* `predictions.csv` → Predicted sales values for test data
* Visualizations and feature analysis are included in the notebooks.
* Random Forest generally provides the highest accuracy (~R² = 0.85).



