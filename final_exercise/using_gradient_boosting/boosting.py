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

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

model = GradientBoostingClassifier(n_estimators=200)
strat = StratifiedKFold(n_splits=3)
result = cross_val_score(model, x, y, cv=strat, n_jobs=-1)

print(result.mean())