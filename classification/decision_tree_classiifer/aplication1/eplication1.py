import pandas as pd
from sklearn.datasets import load_iris

data = load_iris()

x = pd.DataFrame(data.data, columns=[data.feature_names])
y = pd.Series(data.target)

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

strat = StratifiedKFold(n_splits=5)

model = DecisionTreeClassifier()
result = cross_val_score(model, x, y, cv=strat)

print(f'The score of the Decision Tree Classifier was: {result.mean()}')

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

model.fit(x, y)

plt.figure(figsize=(10,8), dpi=150)
plot_tree(model, feature_names=data.feature_names)
plt.show()