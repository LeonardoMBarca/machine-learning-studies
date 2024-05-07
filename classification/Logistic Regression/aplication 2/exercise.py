import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold

# Load dataset
file = load_breast_cancer()
x = pd.DataFrame(file.data, columns=file.feature_names)  # Features
y = pd.Series(file.target)  # Target

# Define hyperparameters for tuning
param_C = np.array([0.01, 0.1, 0.5, 1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 75, 95, 100])
penalty = ['l1', 'l2']
param = {
    'C': param_C,
    'penalty': ['l2'],
    'solver': ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
}
param_with_l1 = {
    'C': param_C,
    'penalty': penalty,
    'solver': ['liblinear', 'saga']    
}

# Initialize Logistic Regression model
model = LogisticRegression(max_iter=2000, tol=0.01)
strat = StratifiedKFold(n_splits=5)
# GridSearchCV for hyperparameter tuning
grid = GridSearchCV(estimator=model, param_grid=param, cv=strat)
grid1 = GridSearchCV(estimator=model, param_grid=param_with_l1, cv=strat)

# RandomizedSearchCV for hyperparameter tuning
random = RandomizedSearchCV(estimator=model, param_distributions=param, n_iter=102, cv=strat)
random1 = RandomizedSearchCV(estimator=model, param_distributions=param_with_l1, n_iter=68, cv=strat)

# Fit models
grid.fit(x, y)
random.fit(x, y)
grid1.fit(x, y)
random1.fit(x, y)

# Print best results from GridSearchCV and RandomizedSearchCV
print(f'The best result using GridSearchCV was {grid.best_score_}, with the parameter C {grid.best_estimator_.C}, and the penalty {grid.best_estimator_.penalty}, and the best solver is {grid.best_estimator_.solver}')
print(f'The best result using RandomizedSearchCV was {random.best_score_}, with the parameter C {random.best_estimator_.C}, and the penalty {random.best_estimator_.penalty}, and the best solver is {random.best_estimator_.solver}')
print(f'The best result using GridSearchCV1 was {grid1.best_score_}, with the parameter C {grid1.best_estimator_.C}, and the penalty {grid1.best_estimator_.penalty}, and the best solver is {grid1.best_estimator_.solver}')
print(f'The best result using RandomizedSearchCV1 was {random1.best_score_}, with the parameter C {random1.best_estimator_.C}, and the penalty {random1.best_estimator_.penalty}, and the best solver is {random1.best_estimator_.solver}')