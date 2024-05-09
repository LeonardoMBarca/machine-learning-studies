import pandas as pd
import numpy as np
data = '/Users/leona/OneDrive/Documentos/GitHub/machine-learning-studies/final_exercise/recipeData.csv'

df = pd.read_csv(data, encoding='ISO-8859-1')
print(df.shape)
for style, count in df['StyleID'].value_counts().items():
    if count < 1000:
        df = df[df['StyleID'] != style]
        
for i, e in df.isna().sum().items():
    if e > 20000:
        df.drop(i, axis=1, inplace=True)

df.fillna(df['PitchRate'].median(), inplace=True)
for column, has_na in df.isna().any().items():
    if has_na:
        df.fillna(df[column].mean(), inplace=True)

df.drop(columns=['URL', 'Name', 'BeerID', 'Style'], axis=1, inplace=True)

df['SugarScale'] = df['SugarScale'].replace('Specific Gravity', 0)
df['SugarScale'] = df['SugarScale'].replace('Plato', 1)
df = pd.get_dummies(df, columns=['BrewMethod'], dtype=int)

print(df.shape)

x = df.drop(columns='StyleID', axis=1)
y = df['StyleID']

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score, log_loss, f1_score

def models(a, b):    
    x = a
    y = b
    normalize = MinMaxScaler(feature_range=(0, 1))
    x_norm = normalize.fit_transform(x)
    
    strat = StratifiedKFold(n_splits=5)
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, tol=0.1),
        'Gaussian': GaussianNB(),
        'Decision Tree': DecisionTreeClassifier()
    }
    
    results = ['KNeighbors:', cross_val_score(KNeighborsClassifier(), x_norm, y, cv=strat, n_jobs=-1).mean()]
    for name, model in models.items():
        result = cross_val_score(model, x, y, cv=strat, n_jobs=-1)
        results.append(f'Model: {name}, Score: {result.mean()}')
    
    print(results)
models(x, y)

def neighbors(a, b):
    x = a
    y = b
    
    normalize = MinMaxScaler(feature_range=(0, 1))
    x_norm = normalize.fit_transform(x)
    
    strat = StratifiedKFold(n_splits=5)
    
    param_grid = {
        'n_neighbors': np.array([5, 9, 11]),
        'metric': ['minkowski', 'chebyshev'],
        'p': np.array([1, 2, 4, 6])
    }
    
    neighbors = KNeighborsClassifier()
    grid = GridSearchCV(estimator=neighbors, param_grid=param_grid, cv=strat, n_jobs=-1)
    grid.fit(x_norm, y)
    
    print(f'The best score of kneighborsclassifier was: {grid.best_score_}, with the best parameters: {grid.best_params_}')
neighbors(x, y)

def tree(a, b):
    x = a
    y = b
    
    strat = StratifiedKFold(n_splits=5)
    
    param_grid = {
        'min_samples_split': np.array([3, 4, 5, 6, 7, 8, 9]),
        'max_depth': np.array([3, 4, 5, 6]),
        'criterion': ['gini', 'entropy']
    }
    
    tree = DecisionTreeClassifier()
    grid = GridSearchCV(estimator=tree, param_grid=param_grid, cv=strat, n_jobs=-1)
    grid.fit(x, y)
    
    print(f'The best score of DecisionTreeClassifier was: {grid.best_score_}, with the best parameters: {grid.best_params_}')
tree(x, y)