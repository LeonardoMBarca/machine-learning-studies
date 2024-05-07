import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = '/Users/leona/OneDrive/Documentos/GitHub/Machine-Learning-Module-1-studies/classification/column_2C_weka (1).csv'
df = pd.read_csv(data)

print(df.head(), df.isna().sum())

df['class'] = df['class'].replace('Abnormal', 1)
df['class'] = df['class'].replace('Normal', 0)

x = df.drop(columns='class', axis=1)
y = df['class']

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=15)

min_samples_split = np.array([2, 3, 4, 5, 6, 7, 8, 9])
max_depth = np.array([1, 2, 3, 4, 5, 6])
criterion = ["gini", "entropy", "log_loss"]
param_grid = {
    'min_samples_split': min_samples_split,
    'max_depth': max_depth,
    'criterion': criterion
}

model = DecisionTreeClassifier()
grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid.fit(x, y)

print(f'The best score of this model was: {grid.best_score_}. The best parameters was: {grid.best_params_}')

best_model = DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_split=9)
best_model.fit(x_train, y_train)

prediction = best_model.predict(x_test)
matrix = confusion_matrix(y_test, prediction)

print(f'The confusion matrix of this model was: \n{matrix}')

plt.figure(figsize=(10, 8), dpi=150)
plot_tree(best_model, feature_names=x.columns)
plt.show()