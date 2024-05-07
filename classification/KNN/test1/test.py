import pandas as pd
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
x = pd.DataFrame(data.data, columns=[data.feature_names])
y = pd.Series(data.target)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score

normalizer = MinMaxScaler(feature_range=(0, 1))
x_norm = normalizer.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_norm, y, test_size=0.30, random_state=16)
k = KFold(n_splits=5)
strat = StratifiedKFold(n_splits=5)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)
result = model.score(x_test, y_test)
resultk = cross_val_score(model, x_norm, y, cv=k)
resultstrat = cross_val_score(model, x_norm, y, cv=strat)

print(f'Score train_test_split: {result}')
print(f'Score kfold: {resultk.mean()}')
print(f'Score stratifiedkfold: {resultstrat.mean()}')