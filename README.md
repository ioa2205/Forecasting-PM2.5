# Spatiotemporal PM2.5 Forecasting Pipeline

This project implements a machine learning pipeline to forecast PM2.5 air quality levels using spatiotemporal data. It includes comprehensive data preprocessing, feature engineering, and model training.

## Project Goal

To accurately predict hourly PM2.5 concentrations for various sensor locations, considering historical pollution levels, meteorological data (implicitly, via engineered features derived from historical data if available), temporal patterns, and spatial relationships.

## Features

*   **Spatiotemporal-Aware Preprocessing:**
    *   Handles datetime parsing.
    *   Performs robust missing value imputation using temporal interpolation within each sensor's data.
    *   Encodes cyclical temporal features (hour, day, month) using sine/cosine transformations.
    *   Generates temporal lag features and rolling window statistics (mean, standard deviation) for the target variable (PM2.5), calculated per sensor.
    *   Implements spatial clustering (KMeans) to group sensor locations based on their coordinates, creating a spatial context feature.
    *   Standardizes numerical features.
*   **Model Training:**
    *   Utilizes LightGBM, a powerful gradient boosting framework, for its efficiency and performance on tabular data.
    *   Employs a time-based train/validation split to prevent data leakage and ensure realistic performance evaluation.
    *   Uses early stopping during training to optimize model performance and prevent overfitting.
*   **Evaluation & Visualization:**
    *   Evaluates model performance using Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).
    *   Generates plots for:
        *   Actual vs. Predicted PM2.5 values.
        *   LightGBM/XGBoost feature importance.

## Dataset Requirements

The primary input is a CSV file (`synthetic_tashkent_aq_2023_2024.csv` ) with at least the following columns:

*   `datetime`: Timestamp of the observation (e.g., `YYYY-MM-DD HH:MM:SS`).
*   `latitude`: Latitude of the sensor.
*   `longitude`: Longitude of the sensor.
*   `pm2_5` (or `PM2.5`): The target variable, PM2.5 concentration.
*   *(Optional but Recommended)*: `temperature`, `humidity`, `wind_speed`, `wind_direction_deg`. These are used by the feature engineering to create a richer feature set for the model. If not present, the corresponding feature engineering steps for them will be skipped or need adjustment.
*   *(Optional)*: `is_peak_traffic_hour`, `is_weekend` binary flags.



## Setup and Installation

1.  **Create a Python Virtual Environment (Recommended):**
    ```bash
    python -m venv .venv
    # Activate the environment
    # Windows:
    # .venv\Scripts\activate
    # macOS/Linux:
    # source .venv/bin/activate
    ```

2.  **Install Required Libraries:**
    ```bash
    pip install pandas numpy scikit-learn lightgbm matplotlib
    ```

## Running the Pipeline

1.  **Configure Script:**
    *   Open the Python script (`your_script_name.py`).
    *   Modify the `RAW_DATA_FILENAME` variable at the top to point to your dataset.
    *   Adjust other configuration parameters (e.g., `TARGET_COLUMN`, `DATETIME_COLUMN`, `N_LAGS`, `ROLLING_WINDOWS`, `MAX_FORECAST_HORIZON_HOURS`) as needed for your specific dataset and requirements.

2.  **Execute the Script:**
    ```bash
    python your_script_name.py
    ```

The script will perform the following steps:
1.  Load raw data.
2.  Preprocess data and engineer features.
3.  Split data into training and validation sets (and a hold-out test set for evaluating the stream).
4.  Scale numerical features.
5.  Train the LightGBM model.
6.  Perform a standard batch evaluation of the model on a test portion.
7.  Demonstrate the iterative (streaming) forecast for a specified horizon, using data from the end of the validation set as the starting point.
8.  Generate and save evaluation plots.


## Potential Improvements & Extensibility

*   **Integrate External Forecasts:** For longer and more accurate "streaming" forecasts, incorporate actual forecasts for meteorological variables (temperature, wind, etc.) instead of just carrying forward last known values.
*   **Advanced Spatial Features:** Explore more sophisticated spatial modeling, such as graph neural networks (GNNs) if inter-station dependencies are strong, or using distance-based features.
*   **Different Models:** Experiment with other time series models (e.g., ARIMA, Prophet, other deep learning architectures if LSTM is reconsidered) and compare their performance.
*   **Error Analysis:** Perform a more detailed analysis of prediction errors (e.g., by time of day, season, location).
*   **Scalability:** For very large datasets or real-time applications, consider using distributed computing frameworks (e.g., Dask, Spark).
*   **Separate Target Scaling:** Implement separate scaling for the target variable to make inverse transformation straightforward and potentially improve model stability for some algorithms.
