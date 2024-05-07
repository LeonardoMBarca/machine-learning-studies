import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix 

# Load and separate data
data = load_breast_cancer()
x = pd.DataFrame(data.data, columns=data.feature_names)  # Features
y = pd.Series(data.target)  # Target

print(y.value_counts())

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=9)

# Instantiate Logistic Regression model
model = LogisticRegression(C=0.01, penalty='l2', max_iter=2000, tol=0.01)

# Train the model
model.fit(x_train, y_train)

# Calculate accuracy
accuracy = model.score(x_test, y_test)
print(f'Accuracy: {accuracy}')

# Make predictions
predictions = model.predict(x_test)
print(predictions)

# Calculate confusion matrix
confusion = confusion_matrix(y_test, predictions)
print(f'The confusion matrix is divided into True Positive | False Positive || False Negative | True Negative. \n{confusion}')