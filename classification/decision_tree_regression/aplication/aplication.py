import pandas as pd

data = 'regression/Admission_Predict_Ver1.1.csv'
df = pd.read_csv(data)
df.drop(columns='Serial No.', axis=1, inplace=True)

x = df.drop(columns='Chance of Admit', axis=1)
y = df['Chance of Admit']

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score, KFold

k = KFold(n_splits=5)
model = DecisionTreeRegressor()
result = cross_val_score(model, x, y, cv=k)

print(result.mean())

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

model.fit(x, y)

plt.figure(figsize=(10, 8), dpi=150)
plot_tree(model, feature_names=df.columns)
plt.show()