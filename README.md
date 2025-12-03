# Taxi Tip Prediction Project

## Project Description
This notebook demonstrates a machine learning pipeline for predicting taxi tip amounts using the Yellow Taxi Trip Data. The process involves data loading, cleaning, feature engineering (implicitly through dropping datetime columns), model training with XGBoost, and evaluation of the model's performance.

## Data Source
The data used in this project is sourced from a distilled version of the 2023 Yellow Taxi Trip Data, specifically from a file located at `/content/drive/MyDrive/Colab Notebooks/Taxi Data/Distilled_2023_Yellow_Taxi_Trip_Data.txt`.

## Setup
To run this notebook, you will need the following Python libraries:
- `pandas`
- `numpy`
- `xgboost`
- `matplotlib`
- `sklearn`

These libraries can typically be installed using pip if they are not already available in your Colab environment:
```bash
pip install pandas numpy xgboost matplotlib scikit-learn
```

Also, ensure that your Google Drive is mounted to access the data file.

## Steps Performed
1.  **Data Loading**: The taxi trip data is loaded from a CSV file into a pandas DataFrame.
2.  **Data Inspection**: Initial checks for missing values and data types are performed.
3.  **Data Cleaning**: 
    *   Missing values in `airport_fee`, `congestion_surcharge`, and `passenger_count` are filled.
    *   The `store_and_fwd_flag` column is dropped due to redundancy.
    *   Anomalous data points are handled, including filtering out `trip_distance` values outside a reasonable range (0.5 to 100 miles) and removing rows with negative numerical values or `total_amount` outside a sensible range (3.7 to 1000).
4.  **Train/Test Split**: The dataset is shuffled and split into training and testing sets, with `tip_amount` as the target variable and other relevant columns as features.
5.  **Model Training**: An `XGBRegressor` model is initialized and trained on the preprocessed training data.
6.  **Prediction**: The trained model makes predictions on the test set.
7.  **Model Evaluation**: The Mean Squared Error (MSE) is used to evaluate the performance of the model, demonstrating an improvement after data cleaning.

## Results
After initial data cleaning and model training, the Mean Squared Error (MSE) was significantly reduced from `1.08` to `0.88`, indicating a substantial improvement in the model's ability to predict tip amounts.
