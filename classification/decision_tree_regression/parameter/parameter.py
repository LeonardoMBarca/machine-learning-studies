import pandas as pd

data = 'regression/Admission_Predict_Ver1.1.csv'
df = pd.read_csv(data)
df.drop(columns='Serial No.', axis=1, inplace=True)

x = df.drop(columns='Chance of Admit', axis=1)
y = df['Chance of Admit']

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV

min_split = np.array([2, 3, 4, 5, 6, 7])
max_depth = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11])
criterion = ['squared_error', 'absolute_error', 'poisson', 'friedman_mse']
grid_tests = {
    'min_samples_split': min_split,
    'max_depth': max_depth,
    'criterion': criterion
}

model = DecisionTreeRegressor()
grid = GridSearchCV(estimator=model, param_grid=grid_tests, cv=5)
grid.fit(x, y)

print(f'The best score was: {grid.best_score_}. The best parameters was: {grid.best_params_}')

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

model_best = DecisionTreeRegressor(criterion='absolute_error', max_depth=4, min_samples_split=3)
model_best.fit(x, y)


plt.figure(figsize=(10, 8), dpi=150)
plot_tree(model_best, feature_names=x.columns)
plt.show()