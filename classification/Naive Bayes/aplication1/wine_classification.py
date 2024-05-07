import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, cross_val_score

data = 'classification/Naive Bayes/wine/wine_dataset.csv'
df = pd.read_csv(data)

x = df.drop(columns='style', axis=1)
y = df['style']

strat = StratifiedKFold(n_splits=5)

model = GaussianNB()

result = cross_val_score(model, x, y, cv=strat)

print(f'The score was: {result.mean()}')