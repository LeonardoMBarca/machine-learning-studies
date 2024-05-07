import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import ElasticNet

file = 'regression/Admission_Predict_Ver1.1.csv'
df = pd.read_csv(file)

df.drop(columns='Serial No.', axis=1, inplace=True)

for i in df.columns:
    if df[i].dtype == object:
        df.drop(columns=i, axis=1, inplace=True)
        
x = df.drop(columns='Chance of Admit', axis=1)
y = df['Chance of Admit']

tests = {
    'alpha': [0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 3, 4, 5, 10, 15, 20, 30, 40, 50, 80, 100, 150, 250, 500, 1000],
    'l1_ratio': [0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.75, 0.8, 0.9, 0.95, 0.99]
}

model = ElasticNet()
find = RandomizedSearchCV(estimator=model, param_distributions=tests, n_iter=300, cv=5, random_state=15)
find.fit(x, y)

print(f'O melhor resultado foi {find.best_score_}')
print(f'O melhor Alpha foi {find.best_estimator_.alpha}')
print(f'O melhor L1_ratio foi {find.best_estimator_.l1_ratio}')
