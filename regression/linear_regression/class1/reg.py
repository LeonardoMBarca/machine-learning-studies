import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

x, y = make_regression(n_samples=200, n_features=1, noise=30)

model = LinearRegression()
model.fit(x, y)

m = model.coef_ 
b = model.intercept_ 
xreg = np.arange(min(x)-1, max(x)+1, 1) 
yreg = m*xreg-b 

plt.scatter(x, y)
plt.plot(xreg, yreg, color='red')
plt.show()