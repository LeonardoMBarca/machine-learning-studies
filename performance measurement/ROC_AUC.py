import pandas as pd
from sklearn.datasets import load_breast_cancer

# Load the breast cancer dataset
data = load_breast_cancer()

# Create pandas DataFrame for features and pandas Series for target
x = pd.DataFrame(data.data, columns=[data.feature_names])
y = pd.Series(data.target)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=9)

# Initialize and train the logistic regression model
model = LogisticRegression(max_iter=2000, tol=0.01)
model.fit(x_train, y_train)

# Calculate the accuracy of the model on the test set
result = model.score(x_test, y_test)
print(f'Accuracy: {result}')

# Predict probabilities for each class on the test set
predicted_proba = model.predict_proba(x_test)
print(predicted_proba)

import numpy as np

# Predict classes for the test set
predicted_class = model.predict(x_test)

# Create a DataFrame with predicted classes and probabilities
result_df = pd.DataFrame(np.column_stack((predicted_class, predicted_proba)), columns=["Predicted Class", "Probability for Class 0", "Probability for Class 1"])

print(result_df)

# Create a var with only first columns of predicted_proba 
probs = predicted_proba[:,1]

# Import the roc_curve
from sklearn.metrics import roc_curve

# Calculate the TPR, FPR and Thresholds
fpr, tpr, thresholds = roc_curve(y_test, probs)

print(f'TPR: {tpr}. FPR: {fpr}. Thresholds: {thresholds}')

import matplotlib.pyplot as plt

# Show data
plt.scatter(fpr, tpr)
plt.show()

# Import the roc_auc_
from sklearn.metrics import roc_auc_score

# show the auc score
print(roc_auc_score(y_test, probs))