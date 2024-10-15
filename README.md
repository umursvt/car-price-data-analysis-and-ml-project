# Vehicle Price Prediction Project

## Project Overview

This project aims to predict vehicle prices based on several features such as **brand**, **model**, **kilometres**, and **engine size** using different machine learning regression models. Various models were trained, and their performance metrics were evaluated to find the best fit for the data.

## Steps of the Project

### 1. **Data Preprocessing**
   - **Data Cleaning**: Removed or replaced missing values, handled outliers, and cleaned up text data (e.g., converting `'unknown'` motor values to `NaN`, cleaning up `Kilometres` and `Price` columns).
   - **Feature Engineering**: 
     - Created new features such as **car age**, **price per kilometre**, and **region** to enhance model performance.
     - One-hot encoding was applied to categorical features such as **brand**, **city**, and **model**.

### 2. **Model Selection**
Several regression models were tested:
   - **Linear Regression**
   - **Ridge Regression**
   - **Lasso Regression**
   - **Random Forest**
   - **Support Vector Regression**

Each model's performance was evaluated using the following metrics:
   - **R² (R-squared)**: Explains how well the features explain the variance in the target (price). A higher value means better performance.
   - **MAE (Mean Absolute Error)**: The average absolute difference between predicted and actual values. Lower is better.
   - **RMSE (Root Mean Squared Error)**: Measures the average magnitude of the error, penalizing larger errors more than MAE. Lower is better.

### 3. **Model Evaluation**
The following metrics were calculated for each model:

#### Linear Regression
- **R²**: 0.35
- **MAE**: 384,643.47
- **RMSE**: 737,586.91

#### Ridge Regression
- **R²**: 0.35
- **MAE**: 378,158.98
- **RMSE**: 735,006.02

#### Lasso Regression
- **R²**: 0.47
- **MAE**: 201,902.17
- **RMSE**: 663,826.68

#### Random Forest
- **R²**: 0.33
- **MAE**: 204,105.62
- **RMSE**: 744,331.96

#### Support Vector Regression
- **R²**: -0.05 (Negative value indicates poor fit)
- **MAE**: 474,945.41
- **RMSE**: 932,445.84

### 4. **Results Analysis**
- **Lasso Regression** performed the best, achieving an **R²** of 0.47, indicating that 47% of the variance in vehicle prices was explained by the model. It also had the lowest **MAE** and **RMSE**, showing more accurate predictions compared to other models.
- **Support Vector Regression** performed the worst, with a negative **R²** value, which suggests that it did not capture the relationships in the data well.
- Overall, Lasso Regression is recommended for this problem due to its better performance in both **MAE** and **RMSE**.

### 5. **Improvements**
To improve the model further, the following steps could be taken:
   - **Feature Selection**: Additional relevant features could be engineered or selected, such as fuel type, transmission, and detailed equipment information.
   - **Hyperparameter Tuning**: Using cross-validation and grid search to optimize the hyperparameters of models like Ridge, Lasso, and Random Forest.
   - **Ensemble Methods**: Combining models such as Gradient Boosting or XGBoost might help improve performance.

## How to Run the Project

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/vehicle-price-prediction.git
   cd vehicle-price-prediction


Dependencies
Python 3.7+
pandas
scikit-learn
numpy
openpyxl


### Açıklamalar:
1. **Data Preprocessing**: Verilerin temizlenmesi ve yeni özelliklerin eklenmesi adımları açıklanmıştır.
2. **Model Evaluation**: Kullanılan modeller ve bunların performans metrikleri listelenmiştir. Her modelin R², MAE ve RMSE değerleri belirtilmiştir.
3. **Improvements**: Model performansını daha da artırmak için önerilen adımlar listelenmiştir.
4. **How to Run the Project**: Projeyi çalıştırmak için gereken adımlar açıklanmıştır.
5. **Dependencies**: Proje için kullanılan Python kütüphaneleri listelenmiştir.



