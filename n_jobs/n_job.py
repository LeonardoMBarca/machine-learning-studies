import pandas as pd
import time

data = '/Users/leona/OneDrive/Documentos/GitHub/machine-learning-studies/regression/Admission_Predict_Ver1.1.csv'

df = pd.read_csv(data)

df.drop(columns='Serial No.', axis=1, inplace=True)

x = df.drop(columns='Chance of Admit', axis=1)
y = df['Chance of Admit']

from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV

grid_param = {
    'alpha': [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'l1_ratio': [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
}

model = ElasticNet()

start_time = time.time()
grid = GridSearchCV(estimator=model, param_grid=grid_param, cv=5)
grid.fit(x, y)
end_time = time.time()

print("Tempo de execução sem n_jobs=-1:", end_time - start_time)

start_time = time.time()
grid = GridSearchCV(estimator=model, param_grid=grid_param, cv=5, n_jobs=-1)
grid.fit(x, y)
end_time = time.time()

print("Tempo de execução com n_jobs=-1:", end_time - start_time)

print(f'The best parameters of this model were: {grid.best_params_}')
print(f'The best score of this model was: {grid.best_score_}')