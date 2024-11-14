#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 19:15:54 2024

@author: frankbogle
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as dp
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

df = dp.read_csv("/Users/frankbogle/Downloads/HousingData.csv")
print(df.shape)
print(df.head())

# Plt histogram best line of fit and correlation matrix
sns.set(rc={'figure.figsize':(12,9)})
sns.histplot(df['MEDV'], bins=25, kde=True)
plt.show()

correlation_matrix = df.corr().round(2)
sns.heatmap(data=correlation_matrix, annot=True)
plt.show()


# Linear regression with single variable
X_rooms = df['RM']
y_price = df['MEDV']

X_rooms = np.array(X_rooms).reshape(-1, 1)
y_price = np.array(y_price).reshape(-1, 1)

# Create train and test datasets
X_train, X_test, Y_train, Y_test = train_test_split(X_rooms, y_price, test_size = 0.2, random_state = 22)

# Create single independent variable regression model
model = LinearRegression()
model.fit(X_train, Y_train)
print(model.intercept_, model.coef_, model.score(X_train, Y_train))

# Calculate the most important independent features
model = Ridge(alpha = 1e-2).fit(X_train, Y_train)

for i in range(len(model.coef_)):
    print(f"{df.columns[i]:<13}"
          f"{abs(model.coef_[i]):.3}")


