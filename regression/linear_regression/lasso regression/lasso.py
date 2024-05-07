import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

file = 'regression/kc_house_data.csv'
df = pd.read_csv(file)

df.drop(columns=['id', 'date', 'zipcode', 'lat', 'long'], axis=1, inplace=True)

for i in df.columns:
    if df[i].dtype == object:
        df.drop(columns=i, axis=1, inplace=True)
for i in df.columns:
    corr = df['price'].corr(df[i])
    if corr < 0.2:
        df.drop(columns=i, axis=1, inplace=True)

x = df.drop(columns='price', axis=1)
y = df['price']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=30, random_state=14)

model = LinearRegression()
model_lasso = Lasso(alpha=50, max_iter=5000, tol=0.1)

model.fit(x_train, y_train)
model_lasso.fit(x_train, y_train)

result = model.score(x_test, y_test)
result_lasso = model_lasso.score(x_test, y_test)

print(f'Linear: {result}')
print(f'Lasso: {result_lasso}')