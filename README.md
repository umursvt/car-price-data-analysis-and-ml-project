Vehicle Price Prediction Project

Project Overview
This project predicts vehicle prices using features like brand, model, kilometres, and engine size. Several machine learning models were trained and evaluated based on different performance metrics.

Steps of the Project
Data Preprocessing:

Cleaned data (handled missing values, removed outliers, cleaned features like Kilometres and Price).
Created new features (e.g., car age, price per kilometre).
Applied one-hot encoding to categorical features.
Model Selection:

Trained and evaluated multiple models: Linear Regression, Ridge Regression, Lasso Regression, Random Forest, and Support Vector Regression.
Model Evaluation:

Metrics used: R², MAE, RMSE.
Best-performing model: Lasso Regression (R² = 0.47, MAE = 201,902, RMSE = 663,827).
Improvements:

Additional features, hyperparameter tuning, and ensemble models could further improve performance.

git clone https://github.com/yourusername/vehicle-price-prediction.git
cd vehicle-price-prediction

Dependencies
Python 3.7+
pandas, scikit-learn, numpy, openpyxl
