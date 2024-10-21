import joblib
from model import x_test
import pandas as pd


model = joblib.load('models/Linear Regression.pkl')

print('deneme',model.predict(x_test))