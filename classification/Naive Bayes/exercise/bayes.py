import pandas as pd
from sklearn.datasets import load_iris

data = load_iris()
x = pd.DataFrame(data.data, columns=[data.feature_names])
y = pd.Series(data.target)

from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=67)

model = GaussianNB()
model.fit(x_train, y_train)

result = model.score(x_test, y_test)

print(f'The score was: {result}')