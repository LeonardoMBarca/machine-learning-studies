import pandas as pd
from sklearn.datasets import load_iris

data = load_iris()

x = pd.DataFrame(data.data, columns=[data.feature_names])
y = pd.Series(data.target)

from sklearn.feature_selection import SelectKBest, chi2

algorithm = SelectKBest(score_func=chi2, k=2)
x_bests = algorithm.fit_transform(x, y)

print(f'Score: {algorithm.scores_}')
print(f'Result of this transformation: {x_bests}')

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

strat = StratifiedKFold(n_splits=5)

model = DecisionTreeClassifier()
result = cross_val_score(model, x_bests, y, cv=strat)
print(f'The score of this model using just the 2 best variables in the dataset was: {result.mean()}')

result1 = cross_val_score(model, x, y, cv=strat)
print(f'The score of this model using every variables in the dataset was: {result1.mean()}')