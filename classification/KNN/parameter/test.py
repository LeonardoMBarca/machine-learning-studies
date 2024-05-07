import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

x = pd.DataFrame(data.data, columns=[data.feature_names])
y = pd.Series(data.target)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV

normalizer = MinMaxScaler(feature_range=(0, 1))
x_norm = normalizer.fit_transform(x)

k_values = np.array([3, 5, 7, 9, 11])
distance = ['minkowski', 'chebyshev']
p_value = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
grid_tests = {
    'n_neighbors': k_values,
    'metric': distance,
    'p': p_value
}

model = KNeighborsClassifier()

grid = GridSearchCV(estimator=model, param_grid=grid_tests, cv=5)

grid.fit(x_norm, y)

print(f'The best score was {grid.best_score_}. The best k value was {grid.best_estimator_.n_neighbors}. The best metric was {grid.best_estimator_.metric}. The best p value was {grid.best_estimator_.p}')