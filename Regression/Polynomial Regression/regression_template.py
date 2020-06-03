#Polynomial regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values #now X is considered as a matrix of features
y = dataset.iloc[:, 2].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
"""
# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()"""

#Fit the Regression model to the data set

#Predicting the result in Regression Model
y_pred = regressor.predict(6.5)


#Visualising Regression Model
plt.scatter(X, y, color = 'blue')
plt.plot(X, regressor.predict(X), color='red')
plt.title("Truth or Bluff (Regression Model)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

#Visualising Regression Model for higher resolution and smoother curve
X_grid = np.arange(min(X), max(X), 0.1)   #it's a vector
X_grid = X_grid.reshape((len(X_grid),1))   #it's a matrix which we want
plt.scatter(X, y, color = 'blue')
plt.plot(X_grid, regressor.predict(X_grid), color='red')
plt.title("Truth or Bluff (Regression Model)")
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()