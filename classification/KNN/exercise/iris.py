import pandas as pd
import numpy as np
from sklearn.datasets import load_iris

data = load_iris()
x = pd.DataFrame(data.data, columns=[data.feature_names])
y = pd.Series(data.target)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix

normalize = MinMaxScaler(feature_range=(0, 1))
x_norm = normalize.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_norm, y, test_size=0.30, random_state=15)

k_values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
metric = ['minkowski', 'chebyshev']
p_values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
grid_test = {
    'n_neighbors': k_values,
    'metric': metric, 
    'p': p_values
}

model = KNeighborsClassifier()
grid = GridSearchCV(estimator=model, param_grid=grid_test, cv=5)
grid.fit(x_norm, y)
model.fit(x_train, y_train)

prediction = model.predict(x_test)

matriz = confusion_matrix(y_test, prediction)

print(f'The best scord was: {grid.best_score_}.') 
print(f'The best K value was: {grid.best_estimator_.n_neighbors}.')
print(f'The best metric was: {grid.best_estimator_.metric}.')
print(f'The best P value was: {grid.best_estimator_.p}')
print(f'The matriz of this model was: \n{matriz}')