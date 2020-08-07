# Loading packages
from sklearn.datasets import load_boston
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import pandas as pd
import numpy as np

boston = load_boston()

# Boston dataset - a dictionary that contains:
boston.keys()

# Features
boston.feature_names

# Data
boston.data.shape   #506 instances, 13 features.

# Description
print(boston.DESCR)
    # - CRIM     per capita crime rate by town
    # - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
    # - INDUS    proportion of non-retail business acres per town
    # - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
    # - NOX      nitric oxides concentration (parts per 10 million)
    # - RM       average number of rooms per dwelling
    # - AGE      proportion of owner-occupied units built prior to 1940
    # - DIS      weighted distances to five Boston employment centres
    # - RAD      index of accessibility to radial highways
    # - TAX      full-value property-tax rate per $10,000
    # - PTRATIO  pupil-teacher ratio by town
    # - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    # - LSTAT    % lower status of the population
    # - MEDV     Median value of owner-occupied homes in $1000's

# Converting to a Pandas dataframe
data = pd.DataFrame(boston.data, columns=boston.feature_names)

# Adding price to dataframe
data['PRICE'] = boston.target

# Overview of data
data.head()
data.info()   # No nulls in dataset, features of type float.
data.describe()

# Separate data into X and y
X, y = data.iloc[:,:-1], data.iloc[:,-1]

print(X.head())

# Convert to dmatrix for enhanced performance
data_dmatrix = xgb.DMatrix(data=X,label=y)

