# Import the pandas library for file manipulation
import pandas as pd
# Set the maximum number of columns and rows to be displayed
pd.set_option('display.max_columns', 64)
pd.set_option('display.max_row', 64)

# Convert the file into a dataframe
file = 'Logistic Regression/data/Data_train_reduced.csv'
df = pd.read_csv(file)

# Drop unnecessary columns
df.drop(columns='Product', axis=1, inplace=True)
df.drop(columns='Respondent.ID', axis=1, inplace=True)
df.drop(columns='q1_1.personal.opinion.of.this.Deodorant', axis=1, inplace=True)

# Analyze file details
print(df.shape)
print(df.dtypes)
print(df.head())

# Drop all columns that are of object type
for col in df.columns:
    if df[col].dtype == object:
        df.drop(columns=col, axis=1, inplace=True)
        
# Calculate the percentage of null values
percent = df.isnull().sum() / len(df) * 100
# Replace null values with the median
for col in df.columns:
    if percent[col] > 20:
        df.drop(columns=col, axis=1, inplace=True)
    elif 0 < percent[col] <= 20:
        df[col] = df[col].fillna(df[col].median())
        
# Create training and testing data
x = df.drop(columns='Instant.Liking', axis=1)
y = df['Instant.Liking']

# Import necessary libraries
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Create parameter values to be tested
param_C = np.array([0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 1, 3, 5, 10, 15, 30, 50, 100])
penalty = ['l1', 'l2']
param_grid = {'C': param_C, 'penalty': penalty}

# Create and train the model using GridSearchCV
model = LogisticRegression(max_iter=2000, tol=0.01)
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5)
grid_search.fit(x, y)

# Print the best result for each parameter
print(grid_search.best_score_)
print(grid_search.best_estimator_.penalty)
print(grid_search.best_estimator_.C)