import pandas as pd
from sklearn.datasets import load_iris

data = load_iris()
x = pd.DataFrame(data.data, columns=[data.feature_names])
y = pd.Series(data.target)

import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

min_split = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
max_depth = np.array([1, 2, 3, 4, 5, 6])
metric = ['gini', 'entropy']
grid_tests = {
    'min_samples_split': min_split,
    'max_depth': max_depth,
    'criterion': metric
}

model = DecisionTreeClassifier()

grid = GridSearchCV(estimator=model, param_grid=grid_tests, cv=5)
grid.fit(x, y)

print(f'The best score of this model was: {grid.best_score_}. The best parameters of this model was: {grid.best_params_}')

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

best_model = DecisionTreeClassifier(criterion='gini', max_depth=3, min_samples_split=3)
best_model.fit(x, y)

plt.figure(figsize=(10, 8), dpi=150)
plot_tree(best_model, feature_names=data.feature_names)
plt.show()