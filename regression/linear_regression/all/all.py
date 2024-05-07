import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
file_path = 'Admission_Predict_Ver1.1.csv'
df = pd.read_csv(file_path)

# Drop unnecessary column
df.drop(columns='Serial No.', axis=1, inplace=True)

# Split data into features (x) and target (y)
x = df.drop(columns='Chance of Admit', axis=1)
y = df['Chance of Admit']

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=30, random_state=14)

# Define a function to train multiple regression models and evaluate their performance
def models(a, b, c, d):
    from sklearn.linear_model import LinearRegression
    from sklearn.linear_model import Ridge
    from sklearn.linear_model import Lasso
    from sklearn.linear_model import ElasticNet 
    
    # Assign training and testing data
    x_train = a
    x_test = b
    y_train = c
    y_test = d
    
    # Initialize regression models
    linear = LinearRegression()
    ridge = Ridge()
    lasso = Lasso()
    elasticnet = ElasticNet()
    
    # Train models
    linear.fit(x_train, y_train)
    ridge.fit(x_train, y_train)
    lasso.fit(x_train, y_train)
    elasticnet.fit(x_train, y_train)
    
    # Evaluate model performance
    scores = {
        'Linear Regression': linear.score(x_test, y_test),
        'Ridge Regression': ridge.score(x_test, y_test),
        'Lasso Regression': lasso.score(x_test, y_test),
        'Elastic Net Regression': elasticnet.score(x_test, y_test)
    }
    best_model = max(scores, key=scores.get)
    
    # Print the best performing model and its score
    print(f'The best performing model was {best_model}, achieving a score of {scores[best_model]}')
    print(scores)

# Call the models function with the training and testing data
models(x_train, x_test, y_train, y_test)
