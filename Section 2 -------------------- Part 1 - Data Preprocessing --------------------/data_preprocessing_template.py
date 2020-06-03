#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Data Preprocessing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Data.csv')  #may change the 'data' val
X = dataset.iloc[:, :-1].values   #independent variables-matrix: Country, Age, Salary
y = dataset.iloc[:, 3].values     #dependent variable-matrix: Purchased

"""
#-----------------------won't really need it in the future

#taking care of missing values
from sklearn.impute import  SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean', verbose=0)
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#encoding the categorical data
#independent needs OneHotEncoder
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
ct_x = ColumnTransformer([("Country", 
                         OneHotEncoder(), [0])],
                        remainder = 'passthrough')
X = np.array(ct_x.fit_transform(X.tolist()), dtype=np.float)

#Dependent doesn't need a OneHotEncoder 
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

------------------------------------------------------------------------
"""

#Splitting the dataset into training set and the testing set

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

""""
# ---------------------may need sometime in the future----
#Feature Scaling
"""
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""


