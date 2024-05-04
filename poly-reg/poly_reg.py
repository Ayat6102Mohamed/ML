import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt



# Load data
data = pd.read_csv('E:/college 4/0semester8/machinee/lab4/assignment2dataset.csv')

# remove rows with non values
data.dropna(how='any', inplace=True)

# Preprocessing --  feature Encoding
data['Extracurricular Activities'] = data['Extracurricular Activities'].map({'yes': 1, 'no': 0})

# # Feature Selection -- according to 2-feature with high correlation with the "Performance Index" target
corr = data.corr()
top_features = corr['Performance Index'].nlargest(3).index[1:]
print(f"Top features selected: {top_features}")
X = data[top_features]
Y = data['Performance Index']


# Split the data into 80%training and 20%testing
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, shuffle=True, random_state=10)

def polynomial_transformation(X, degree):
    X_poly=np.column_stack([X**i for i in range(0, degree + 1)])
    return X_poly


# degree = int(input("Enter the degree of the polynomial: "))

def polynomial_regression(X_train, y_train, X_test, y_test, degree):
     X_train_poly = polynomial_transformation(X_train, degree)
     X_test_poly = polynomial_transformation(X_test, degree)

     # Fit new_features by Linear Regression
     poly_model = linear_model.LinearRegression()
     poly_model.fit(X_train_poly, y_train)

     # Predictions
     y_train_predict = poly_model.predict(X_train_poly)
     y_test_predict = poly_model.predict(X_test_poly)

     # MSE
     mse_train = metrics.mean_squared_error(y_train, y_train_predict)
     mse_test = metrics.mean_squared_error(y_test, y_test_predict)
     return mse_train,mse_test

# MSE for degrees 2, 3,4..
mse_train_lst = []
mse_test_lst = []
deg=range(2,9)

for degree in deg:
     mse_train,mse_test = polynomial_regression(X_train, y_train, X_test, y_test, degree)
     mse_train_lst.append(mse_train)
     mse_test_lst.append(mse_test)
     print(f"Degree {degree}: MSE_TRAIN,TEST = {mse_train,mse_test}")

# Plotting
plt.plot(deg, mse_train_lst, label='Train MSE')

plt.plot(deg, mse_test_lst, label='Test MSE')
plt.xlabel('Degree')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('Polynomial Regression: Train Vs Test MSE')
plt.show()
