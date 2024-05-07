import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()
x = pd.DataFrame(data.data, columns=[data.feature_names])
y = pd.Series(data.target)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

normalize = MinMaxScaler(feature_range=(0, 1))
x_norm = normalize.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_norm, y, test_size=0.30, random_state=15)

model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)

result = model.score(x_test, y_test)
print(f'Score: {result}')

prediction = model.predict(x_test)

matriz = confusion_matrix(y_test, prediction)

print(f'matriz: \n{matriz}')