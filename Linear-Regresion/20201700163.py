import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

# Loading data
data = pd.read_csv('E:/college 4/0semester8/machinee/assignment/Assigment1_ML/assignment1dataset.csv')
print(data.describe())


X = data['Hours Studied']
X2 = data['Previous Scores']
X3 = data['Sleep Hours']
X4=data['Sample Question Papers Practiced']
# data['Combined Feature'] = data['Hours Studied'] * data['Sleep Hours']
# X5 = data['Combined Feature']
data['New Feature'] = data['Hours Studied'] ** 2
X_new_feature = data['New Feature']
Y = data['Performance Index']



# gradientdescent for each variable
def gradient_descent(X, y):
    L = 0.0001
    epochs = 1000   # The number of iterations to perform gradient descent
    m = 0
    c = 0
    n = float(len(X))
    for _ in range(epochs):
        Y_pred = m * X + c
        D_m = (-2/n) * sum((y - Y_pred) * X)
        D_c = (-2/n) * sum(y - Y_pred)
        m = m - L * D_m
        c = c - L * D_c
    return m, c
# Update m,c
m_hours_studied, c_hours_studied = gradient_descent(X, Y)
m_previous_scores, c_previous_scores = gradient_descent(X2, Y)
m_sleep_hours, c_sleep_hours = gradient_descent(X3, Y)
m_Sample_Question_Papers, c_Sample_Question_Papers = gradient_descent(X4, Y)
#m_New_Feature, c_New_Feature = gradient_descent(X5, Y)
m2_New_Feature, c2_New_Feature = gradient_descent(X_new_feature, Y)


# Predictions for each variable
prediction_hours_studied = m_hours_studied * X + c_hours_studied
prediction_previous_scores = m_previous_scores * X2 + c_previous_scores
prediction_sleep_hours = m_sleep_hours * X3 + c_sleep_hours
prediction_Sample_Question_Papers = m_Sample_Question_Papers * X4 + c_Sample_Question_Papers
#prediction_new = m_New_Feature * X5 + c_New_Feature
prediction_new2 = m2_New_Feature * X_new_feature + c2_New_Feature

# Print Mean Squared Error for each variable
print('Mean Squared Error for Previous Scores:', metrics.mean_squared_error(Y, prediction_previous_scores))
print('Mean Squared Error for Hours Studied:', metrics.mean_squared_error(Y, prediction_hours_studied))
print('Mean Squared Error for Sleep Hours:', metrics.mean_squared_error(Y, prediction_sleep_hours))
print('Mean Squared Error for Sample Question Papers:', metrics.mean_squared_error(Y, prediction_Sample_Question_Papers))
#print('Mean Squared Error for New Feature:', metrics.mean_squared_error(Y, prediction_new))
print('Mean Squared Error for New Feature:', metrics.mean_squared_error(Y, prediction_new2))

# Plotting
plt.figure(figsize=(20, 5))
plt.subplot(1, 5, 1)
plt.scatter(X, Y)
plt.xlabel('Hours Studied', fontsize=15)
plt.ylabel('Performance Index', fontsize=15)
plt.plot(X, prediction_hours_studied, color='red', linewidth=3)
plt.subplot(1, 5, 2)
plt.scatter(X2, Y)
plt.xlabel('Previous Scores', fontsize=15)
plt.ylabel('Performance Index', fontsize=15)
plt.plot(X2, prediction_previous_scores, color='red', linewidth=3)
plt.subplot(1, 5, 3)
plt.scatter(X3, Y)
plt.xlabel('Sleep Hours', fontsize=15)
plt.ylabel('Performance Index', fontsize=15)
plt.plot(X3, prediction_sleep_hours, color='red', linewidth=3)
plt.subplot(1, 5, 4)
plt.scatter(X4, Y)
plt.xlabel('Sample Question Papers Practiced', fontsize=15)
plt.ylabel('Performance Index', fontsize=15)
plt.plot(X4, prediction_Sample_Question_Papers, color='red', linewidth=3)
plt.subplot(1, 5, 5)
plt.scatter(X_new_feature, Y)
plt.xlabel('Combined Feature', fontsize=15)
plt.ylabel('Performance Index', fontsize=15)
plt.plot(X_new_feature, prediction_new2, color='red', linewidth=3)
plt.show()