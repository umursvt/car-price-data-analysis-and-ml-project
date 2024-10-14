import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import locale

"""
Bu projede makinme öğrenimi ile tarih kolonunu da bir feature olarak kullanmak istedim modeli eğitirken ama malesef tarih değerlerinin hepsi
10.aya(Ekim) ait olması gereği ile buradan bir categorik veri getirmek anlamsız geldi. yine de yaptığım işleri silmemek için kodları tutuyorum burada
feature datasetinde(X) tarih bilgisine yer vermeyeceğim

"""

# Tarih kolonundaki  str verileri date e çevirmek için gerekli
locale.setlocale(locale.LC_TIME, 'tr_TR.UTF-8')

dataset_suv = pd.read_csv('web_scrapping and car_datasets/car_dataset_arabam_com_suv.csv')
dataset_oto = pd.read_csv('web_scrapping and car_datasets/car_dataset_arabam_com_otomobil.csv')

# Concat datasets
all_dataset = pd.concat([dataset_suv,dataset_oto],axis=0)

# Str formatındaki verileri datetime a çevir
all_dataset['Advert Date'] = pd.to_datetime(all_dataset['Advert Date'], format='%d %B %Y', errors='coerce')

# Boş değerleri belirli bir tarih ile doldurun Örn:2024-10-14
all_dataset['Advert Date'] = all_dataset['Advert Date'].fillna(pd.Timestamp("2024-10-14"))

# Yıl ve ay bilgisini kolona ekle
all_dataset['Year'] = all_dataset['Advert Date'].dt.year
all_dataset['Month'] = all_dataset['Advert Date'].dt.month
all_dataset['Day'] = all_dataset['Advert Date'].dt.day

# City kolonda sadece şehir bilgisini tut
all_dataset['City'] = all_dataset['City'].str.split('\n').str[0]
# Motor kolonunda sayısal veri tut gerisini at




# Lambda fonksiyonu ile motor hacmi olup olmadığını kontrol et
all_dataset['Motor'] = all_dataset['Motor'].apply(
    lambda x: x.split(' ')[0] if x.split(' ')[0].replace('.', '', 1).isdigit() else 'unknown')

all_dataset['Motor'] = all_dataset['Motor'].apply(
    lambda x: str(float(x)/100) if x.replace('.', '', 1).isdigit() and float(x) >= 100 
    else str(float(x)/10) if x.replace('.', '', 1).isdigit() and float(x) >= 10 
    else x
)

# Feature and Target
X = all_dataset.drop(['Advert Date', 'Price', 'Month', 'Day'],axis=1)
Y = all_dataset['Price']

print(all_dataset)

categorical_columns = ['City', 'Brand', 'Model','Motor']  


# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_columns) 
    ])

# X datasetini dönüştür
X_preprocessed = preprocessor.fit_transform(X)
