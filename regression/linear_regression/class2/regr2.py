import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

x, y = make_regression(n_samples=200, n_features=1, noise=30) 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=30) 

model = LinearRegression()
model.fit(x_train, y_train)

m = model.coef_
b = model.intercept_
regx = np.arange(min(x_train)-1, max(x_train)+1, 1)
regy = m*regx+b
resultado = model.score(x_test, y_test)

plt.scatter(x_train, y_train)
plt.plot(regx, regy, color='red')
plt.title('Train')
plt.show()

plt.scatter(x_test, y_test)
plt.plot(regx, regy, color='red')
plt.title('Test')
plt.show()

print(f'Score: {resultado}')