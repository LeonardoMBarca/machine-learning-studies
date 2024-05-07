import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

file = 'regression/kc_house_data.csv'
df = pd.read_csv(file)

df.drop(columns=['id', 'date', 'zipcode', 'lat', 'long'], axis = 1, inplace=True)

x = df.drop(columns='price', axis=1)
y = df['price']

modelo = LinearRegression()

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=10)

modelo.fit(x_train, y_train)
result = modelo.score(x_test, y_test)
print(result)