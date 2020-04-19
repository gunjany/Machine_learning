# Applying SLR for wine quality prediction
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing dataset
dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
data = pd.read_csv(dataset_url, sep=';')
print(data.head())

# indedpendent and dependent set
X = data.iloc[:, :-1].values
y = data.iloc[:, 11].values
# Splitting dataset into training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size = 0.20,
                                                    random_state = 0,
                                                    stratify = y)

# Applying feature scaling on X
from sklearn.preprocessing import StandardScaler
X_sc = StandardScaler()
X_train = X_sc.fit_transform(X_train)
X_test = X_sc.fit_transform(X_test)

# Fitting Model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression().fit(X_train, y_train)

#Prediction the test set result
y_pred = regressor.predict(X_test)
y_pred = np.round(y_pred)
y_pred = y_pred.astype(int)

# Making the optimal model using backward elimination
import statsmodels.regression.linear_model as sm
X = np.append(arr = np.ones((1599, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]]
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
regressor_ols.summary()

X_opt = X[:, [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11]]
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
regressor_ols.summary()

X_opt = X[:, [0, 1, 2, 3, 5, 6, 7, 9, 10, 11]]
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
regressor_ols.summary()

X_opt = X[:, [0, 2, 5, 6, 7, 9, 10, 11]]
regressor_ols = sm.OLS(endog = y, exog = X_opt).fit()
regressor_ols.summary()

# Training the model
# Once the optimal columns that have better significant effect on the
# dependent variable, make and train the new model to the machine by incuding
# only those best-fit columns!

# Splitting the dataset into optimal training and set set
from sklearn.model_selection import train_test_split
X_opt_train, X_opt_test, y_opt_train, y_opt_test = train_test_split(X_opt, y, 
                                                    test_size = 0.20,
                                                    random_state = 0,
                                                    stratify = y)

# Fitting the new model
from sklearn.linear_model import LinearRegression
regressor_opt = LinearRegression().fit(X_opt_train, y_opt_train)

# Predicting the optimal result
y_pred_ols = regressor_opt.predict(X_opt_test)
y_pred_ols = np.round(y_pred_ols)
y_pred_ols = y_pred_ols.astype(int)

#Checking accuracy of the model
from sklearn.metrics import confusion_matrix
cf = confusion_matrix(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred_ols)

from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test, y_pred))
print(mean_squared_error(y_test, y_pred_ols))