#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 08:23:49 2024

@author: frankbogle
"""

import matplotlib.pyplot as plt
import pandas as dp
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression

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

# Another take on correlation matrix
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot = True, cmap = 'coolwarm')
plt.show()

# Drop na values from df, perhaps imputing values would be better
df = df.dropna()
print(df.shape)


# Create subsets of df containing dependent and independent variables
X = df[['RM', 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'TAX', 'PTRATIO', 'DIS', 'RAD', 'B', 'LSTAT']]  # Example features
y = df['MEDV']

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)

# Train Ridge regression model to determine feature importance
ridge_model = Ridge(alpha=1e-2)
ridge_model.fit(X_train, y_train)

# Display intercept, coefficients, and model score
print("Intercept:", ridge_model.intercept_)
print("Coefficients:", ridge_model.coef_)
print("Model R^2 Score:", ridge_model.score(X_train, y_train))

# Display each feature with its importance score
feature_importance = abs(ridge_model.coef_)
feature_names = X.columns

# Zip feature names and importance, then sort by importance in descending order
sorted_features = sorted(zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True)

# Print sorted features with their importance scores
for feature, importance in sorted_features:
    print(f"{feature:<10}: {importance:.3f}")

# Use SelectKBest algorithm to select best 4 features
selector = SelectKBest(score_func = f_regression, k = 4)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Train model on selected features
model = LinearRegression()
model.fit(X_train_selected, y_train)

# View selected features
selected_features = X.columns[selector.get_support(indices=True)]
print("Selected Features:", selected_features)

# View selected features
predictions = model.predict(X_test_selected)
sorted_predictions = sorted(predictions.flatten(), reverse = True)
print("Predictions in descending order:", sorted_predictions)

# Evaluate model
print("R^2 Score: ", model.score(X_test_selected, y_test))


    

