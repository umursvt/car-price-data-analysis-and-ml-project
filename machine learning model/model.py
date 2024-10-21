import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import locale
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score,mean_squared_error
import numpy as np
import os


    
"""
This project aims to apply machine learning to predict car prices based on various features such as the car's brand, model, age, mileage, motor capacity, and more. Initially, the dataset included date information, but since all dates fall within the same month (October), including the date as a categorical feature was deemed irrelevant. However, the code for handling the date feature has been retained for reference purposes.

Key steps in the project include:
1. Data preprocessing: Converting date strings to datetime objects, extracting year, month, and day from the date, and handling missing values.
2. Cleaning categorical columns such as 'City', 'Motor', and 'Kilometres' by applying string operations, removing unnecessary characters, and filling missing values.
3. Calculating derived features such as the car's age and price per kilometer.
4. Preprocessing the features by applying OneHotEncoder for categorical data and normalizing numerical data.
5. Training and evaluating multiple regression models, including Linear Regression, Ridge, Lasso, Random Forest, and Support Vector Regression (SVR).
6. For each model, performance metrics such as R², Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) are calculated and saved to an Excel file.

Note: The project excludes the 'Advert Date' feature from the final model training due to the homogeneity of date values in the dataset.
"""
# Tarih kolonundaki  str verileri date e çevirmek için gerekli
locale.setlocale(locale.LC_TIME, 'tr_TR.UTF-8')

if not os.path.exists('models'):
    os.makedirs('models')
# Bilimsel gösterimi devre dışı bırak
pd.set_option('display.float_format', '{:.2f}'.format)

dataset_suv = pd.read_csv('web_scrapping and car_datasets/car_dataset_arabam_com_suv.csv')
dataset_oto = pd.read_csv('web_scrapping and car_datasets/car_dataset_arabam_com_otomobil.csv')

# Concat datasets
all_dataset = pd.concat([dataset_suv,dataset_oto],axis=0)

# Str formatındaki verileri datetime a çevir
all_dataset['Advert Date'] = pd.to_datetime(all_dataset['Advert Date'], format='%d %B %Y', errors='coerce')

# Boş değerleri belirli bir tarih ile doldurun Örn:2024-10-14
all_dataset['Advert Date'] = all_dataset['Advert Date'].fillna(pd.Timestamp("2024-10-14"))

# Yıl ve ay bilgisini kolona ekle
all_dataset['Advert Year'] = all_dataset['Advert Date'].dt.year
all_dataset['Advert Month'] = all_dataset['Advert Date'].dt.month
all_dataset['Advert Day'] = all_dataset['Advert Date'].dt.day

# Araç Yaşı heaspla
all_dataset['Car Age'] = 2024 - all_dataset['Year']



# City kolonda sadece şehir bilgisini tut
all_dataset['City'] = all_dataset['City'].str.split('\n').str[0]
all_dataset['City'] = all_dataset['City'].astype(object).apply(
    lambda x: x.replace('\r','').strip() if pd.notnull(x) else 'unknown'
)


# Lambda fonksiyonu ile motor hacmi olup olmadığını kontrol et
all_dataset['Motor'] = all_dataset['Motor'].apply(
    lambda x: x.split(' ')[0] if x.split(' ')[0].replace('.', '', 1).isdigit() else 'unknown')

all_dataset['Motor'] = all_dataset['Motor'].apply(
    lambda x: str(float(x)/100) if x.replace('.', '', 1).isdigit() and float(x) >= 100 
    else str(float(x)/10) if x.replace('.', '', 1).isdigit() and float(x) >= 10 
    else x
)
# Unknown motor hacmi verilerini doldur
all_dataset['Motor'] = all_dataset['Motor'].replace('unknown', np.nan).astype(float)



# Kilometre kolonundaki '-' ve '.' karakterlerini temizle
all_dataset['Kilometres'] = all_dataset['Kilometres'].astype(str).apply(
    lambda x: (x.replace('.', '').replace('-', '').strip()) if pd.notnull(x) and x.strip() != '' else 0
)
# kolonda '' olandeğerleri 0 ile değiştirdim
all_dataset['Kilometres'] = all_dataset['Kilometres'].astype(str).apply(
    lambda x: 0 if x=='' else x
)
all_dataset['Kilometres'] = all_dataset['Kilometres'].astype(int)



all_dataset['Price'] = all_dataset['Price'].astype(str).apply(
    lambda x: (x.replace('.','').strip())  if pd.notnull(x) else np.nan
)

all_dataset['Price'] = all_dataset['Price'].astype(int)

# # Kilometre başına fiyat hesaplama (Price_per_Km)
# all_dataset['Price_per_Km'] = all_dataset['Price'] / (all_dataset['Kilometres'] + 1) 


zero_kilometres_rows = all_dataset.loc[all_dataset['Kilometres'] == 0] 


# Feature and Target
X = all_dataset.drop([ 'Price','Advert Year','Advert Month','Advert Day','Year','Advert Date','Unnamed: 0',],axis=1)
Y = all_dataset['Price']

print(X)

categorical_columns = ['City', 'Brand', 'Model','Kilometres','Colors']  
numerical_columns = ['Kilometres','Car Age','Motor']

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_columns),
        ('num',OneHotEncoder(),numerical_columns)
     
    ])

# X datasetini dönüştür
X_preprocessed = preprocessor.fit_transform(X)

x_train,x_test,y_train,y_test = train_test_split(X_preprocessed,Y,test_size=0.1,random_state=42)

# models
models ={
    'Linear Regression': LinearRegression(),
    'Ridge Regression':Ridge(),
    'Lasso Regression': Lasso(),
    'Random Forest': RandomForestRegressor(),
    'Support Vector Regression':SVR()
}

with pd.ExcelWriter('results.xlsx') as writer:
    for name,model in models.items():
        model.fit(x_train,y_train)
        y_pred = model.predict(x_test)
        
        # Metrik değer hesapları
        r2 = r2_score(y_test,y_pred)
        mae =mean_absolute_error(y_test,y_pred)

        metrics_df =pd.DataFrame({
            'Metrics':['R2','MAE'],
            'Values':[r2,mae]
        })

        results_df = pd.DataFrame({
                'Real Value': y_test,
                'Predicted Value': y_pred
            })
        print(results_df)
        joblib.dump(preprocessor, f"models/{name}.pkl")
        joblib.dump(model,f"models/{name}.pkl")
        
        print(f"{name} başarıyla kaydedildi")
        
        # tahmin için yeni veriler
        new_data = pd.DataFrame({
            'City': ['Edirne'],  
            'Brand': ['Citroen'],       
            'Model': ['C3'], 
            'Motor': [1.4], 
            'Kilometres':[260000],
            'Colors': ['Bej'],    
            'Car Age': [16],                 
        })
        new_data_preprocessed = preprocessor.transform(new_data)

        new_value_predict=model.predict(new_data_preprocessed)
        
        new_value_predicted_df = pd.DataFrame({
            'City': ['Edirne'],  
            'Brand': ['Citroen'],       
            'Model': ['C3'], 
            'Motor': [1.4], 
            'Kilometres':[260000],
            'Colors': ['Bej'],    
            'Car Age': [16],
            'Predicted Value' : new_value_predict
        })
        
        print('-'*50)
        results_df.to_excel(writer,sheet_name=name,index=False)
        new_value_predicted_df.to_excel(writer,sheet_name=name,startcol=len(results_df.columns)+2,index=False)
        metrics_df.to_excel(writer,sheet_name=name,startcol=len(results_df.columns)+len(new_value_predicted_df.columns)+2,index=False)
