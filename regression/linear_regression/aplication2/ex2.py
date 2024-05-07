import pandas as pd
from sklearn.model_selection import train_test_split

arquivo = 'regression/Admission_Predict_Ver1.1.csv'
df = pd.read_csv(arquivo)
for i in df.columns:
    if df[i].dtype != float:
        df[i] = df[i].astype(float)
for i in df.columns:
    corr = df['Chance of Admit'].corr(df[i])
    if corr < 0.2:
        df.drop(columns=i, axis=1, inplace=True)
        
x = df.drop(columns='Chance of Admit')
y = df['Chance of Admit']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=30, random_state=14)

def models(a, b, c, d):
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import Ridge
    from sklearn.linear_model import Lasso
    from sklearn.linear_model import ElasticNet
    
    x_train = a
    x_test = b
    y_train = c
    y_test = d
    
    model = LinearRegression()
    model_ridge = Ridge()
    model_lasso = Lasso()
    model_elasticnet = ElasticNet()
    
    model.fit(x_train, y_train)
    model_ridge.fit(x_train, y_train)
    model_lasso.fit(x_train, y_train)
    model_elasticnet.fit(x_train, y_train)
    
   
    result = model.score(x_test, y_test),
    result_ridge = model_ridge.score(x_test, y_test),
    result_lasso = model_lasso.score(x_test, y_test),
    result_elasticnet = model_elasticnet.score(x_test, y_test)
    dict = {"Linear Regression": result, "Ridge Regression": result_ridge, "Lasso Regression": result_lasso, "ElsasticNet Regression": result_elasticnet}
    best_score = max(dict, key=dict.get)
    print(dict)
    print(f'Best model: {best_score}. Score: {dict[best_score]}')
models(x_train, x_test, y_train, y_test)