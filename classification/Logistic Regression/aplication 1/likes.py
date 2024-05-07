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

# Analyze the details of the file
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
  
# Create X and Y variables      
x = df.drop(columns='Instant.Liking', axis=1)
y = df['Instant.Liking']

# Import necessary functions from scikit-learn library
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold

# Initialize the logistic regression model
model = LogisticRegression(max_iter=2000, tol=0.01)
# Initialize the StratifiedKFold function for cross-validation
stratifiedkfold = StratifiedKFold(n_splits=5)
# Calculate the result based on the cross-validation model
result = cross_val_score(model, x, y, cv=stratifiedkfold)

# Print the mean of the results
print(result.mean())