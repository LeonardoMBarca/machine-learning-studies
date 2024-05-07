import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

file = 'regression/Admission_Predict_Ver1.1.csv'
df = pd.read_csv(file)

df.drop(columns='Serial No.', axis=1, inplace=True)

x = df.drop(columns='Chance of Admit', axis=1)
y = df['Chance of Admit']

def models(a, b):
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import Ridge
    from sklearn.linear_model import Lasso
    from sklearn.linear_model import ElasticNet
    
    x = a
    y = b 
    
    tests_ridge_lasso = {
        'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 5, 7, 8, 10, 12, 15, 20, 30, 40, 100, 200, 250]
    }
    tests_elasticnet = {
        'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 80, 100, 150, 250, 500, 1000],
        'l1_ratio': [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95, 0.99]
    }
    
    linear = LinearRegression()
    ridge = Ridge()
    lasso = Lasso()
    elasticnet = ElasticNet()
    
    kfold = KFold(n_splits=5)
    result_linear = cross_val_score(linear, x, y, cv=kfold)
    
    find_ridge = RandomizedSearchCV(estimator= ridge, param_distributions=tests_ridge_lasso, n_iter=300, cv=5, random_state=15)
    find_lasso = RandomizedSearchCV(estimator =lasso, param_distributions=tests_ridge_lasso, n_iter=300, cv=5, random_state=15)
    find_elasticnet = RandomizedSearchCV(estimator =elasticnet, param_distributions=tests_elasticnet, n_iter=300, cv= 5, random_state=15)
    
    find_ridge.fit(x, y)
    find_lasso.fit(x, y)
    find_elasticnet.fit(x, y)
    
    list = {
        'Regressão Linear': result_linear.mean(),
        'Regressão Ridge': find_ridge.best_score_,
        'Regressão Lasso': find_lasso.best_score_,
        'Regressão Elastic Net': find_elasticnet.best_score_
    }
    maxi = max(list, key=list.get)
    print(f'Best Model: {maxi}. Score: {list[maxi]}')
    print(list)

models(x, y)