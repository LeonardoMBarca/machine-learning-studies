import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge

file = '/Users/leona/OneDrive/Documentos/GitHub/machine-learning-studies/regression/Admission_Predict_Ver1.1.csv'
df = pd.read_csv(file)
df.drop(columns='Serial No.', axis=1, inplace=True)

x = df.drop(columns='Chance of Admit', axis=1)
y = df['Chance of Admit']

model = Ridge()

rfe = RFE(estimator=model, n_features_to_select=5)
fit = rfe.fit(x, y)

print(f'Number of features: {fit.n_features_}')
print(f'Selected features: {fit.support_}')
print(f'Feature Rankings: {fit.ranking_}')

from sklearn.tree import DecisionTreeRegressor

model1 = DecisionTreeRegressor()
rfe1 = RFE(estimator=model1, n_features_to_select=5)
fit1 = rfe1.fit(x, y)

print(f'Number of features: {fit1.n_features_}')
print(f'Selected features: {fit1.support_}')
print(f'Feature Rankings: {fit1.ranking_}')