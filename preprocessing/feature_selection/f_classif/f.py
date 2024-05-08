import pandas as pd
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest, f_classif

data = load_iris()

x = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

algorithm = SelectKBest(score_func=f_classif, k=2)
best_datas = algorithm.fit_transform(x, y)

print(f'The scores of columns was: {algorithm.scores_}')
print(f'The best data was: \n{best_datas}')