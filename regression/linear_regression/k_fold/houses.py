import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

file = 'regression/kc_house_data.csv'
df = pd.read_csv(file)

df.drop(columns=['id', 'zipcode','lat','long'], axis=1, inplace=True)

for i in df.columns:
    if df[i].dtype == object:
        df.drop(columns=i, axis=1, inplace=True)
for i in df.columns:
    corr = df['price'].corr(df[i])
    if corr < 0.2:
        df.drop(columns=i, axis=1, inplace=True)

x = df.drop(columns='price', axis=1)
y = df['price']

def models(a, b):
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import Ridge
    from sklearn.linear_model import Lasso
    from sklearn.linear_model import ElasticNet
    
    x = a
    y = b
    
    linear = LinearRegression()
    ridge = Ridge()
    lasso = Lasso()
    elasticnet = ElasticNet()
    
    Kfold = KFold(n_splits=5)
    
    linear_ = cross_val_score(linear, x, y, cv=Kfold)
    ridge_ = cross_val_score(ridge, x, y, cv=Kfold)
    lasso_ = cross_val_score(lasso, x, y, cv=Kfold)
    elasticnet_ = cross_val_score(elasticnet, x, y, cv=Kfold)
    
    list = {
        'Linear': linear_.mean(),
        'Ridge': ridge_.mean(),
        'Lasso': lasso_.mean(),
        'Elastic Net': elasticnet_.mean()
    }
    maxi = max(list, key=list.get)
    print(f'O melhor modelo foi o {maxi}, tendo score de {list[maxi]}')
    print(list)

models(x, y)