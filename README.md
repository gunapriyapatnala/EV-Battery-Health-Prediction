# EV Battery Intelligence & Predictive Health System
Machine Learning-Based State of Health (SoH) and Remaining Useful Life (RUL) Prediction
##Project Overview
This project uses machine learning to analyze Electric Vehicle (EV) battery performance and predict battery health and remaining useful life. It helps understand battery degradation patterns using historical battery data.

## Objectives
- Predict battery State of Health (SoH).
- Estimate Remaining Useful Life (RUL) in cycles.
- Analyze battery degradation patterns.
- Compare machine learning models based on their prediction performance.
- Provide an interactive dashboard for battery health prediction.

## Technologies Used
- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn
- **Machine Learning Algorithms:** Linear Regression, Random Forest, Gradient Boosting
- **Web Interface:** Streamlit
- **Model Persistence:** Joblib

## Input Features
- Cycle Number
- Voltage (V)
- Temperature (°C)
- Capacity (Ah)

## Project Workflow
1. Dataset Collection
2. Data Cleaning and Preprocessing
3. Exploratory Data Analysis (EDA)
4. Feature Engineering and Selection
5. Train-Test Split
6. Machine Learning Model Training
7. Model Evaluation and Comparison
8. Hyperparameter Tuning
9. Model Saving Using Joblib
10. Streamlit Dashboard Development

## Model Evaluation
The machine learning models are compared using:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R² Score

## What I Learned
Through this project, I gained practical experience in data preprocessing, exploratory data analysis, training and evaluating machine learning models, and building an interactive prediction dashboard. I also learned how machine learning can be applied to real-world battery health prediction problems.

## Project Status
Developed as an academic machine learning project. Dataset source and reproducibility instructions are being documented.

## Dataset Information

This project uses a cleaned battery cycle-level dataset
to predict State of Health (SoH) and Remaining Useful Life (RUL).

The dataset includes cycle number, voltage, temperature,
capacity, SoH, and RUL.

The dataset is not included in this repository.
To run the application, place the required CSV file
inside the `Data` folder with the filename
`battery_cycle_level_dataset_CLEAN_FINAL.csv`.

The original dataset source will be added once verified.
