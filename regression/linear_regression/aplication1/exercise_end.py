import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

arquivo = 'regression/kc_house_data.csv'
df = pd.read_csv(arquivo)

for i in df.columns:
    if df[i].dtype == object:
        df.drop(columns = i, axis=1, inplace=True)
for i in df.columns:
    corr = df['price'].corr(df[i])
    if corr < 0.2:
        df.drop(columns=i, axis=1, inplace=True)

x = df.drop(columns='price', axis=1)
y = df['price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=30)

model = LinearRegression()
model.fit(x_train, y_train)

result = model.score(x_test, y_test)
print(result)