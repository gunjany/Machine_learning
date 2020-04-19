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

#Checking accuracy of the model
from sklearn.metrics import confusion_matrix
cf = confusion_matrix(y_test, y_pred)

from sklearn.metrics import mean_squared_error
print(mean_squared_error(y_test, y_pred))