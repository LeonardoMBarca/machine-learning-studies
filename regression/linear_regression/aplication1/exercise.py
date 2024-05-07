import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

file = 'regression/kc_house_data.csv'

df = pd.read_csv(file)
df.drop(columns=['id', 'date', 'zipcode', 'lat', 'long'], axis = 1, inplace=True)
print(df.columns)

x = df.drop(columns='price', axis=1)
y = df['price']

model = LinearRegression()

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.30)

model.fit(x_treino, y_treino)
result = model.score(x_teste, y_teste)
print(result)