#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  9 06:50:58 2024

@author: frankbogle
"""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
import seaborn as sns

df = pd.read_csv("/Users/frankbogle/Downloads/framingham.csv")
print("Framingham shape with na values: ", df.shape)

# check levels of na data
print(df.isna().sum().sort_values(ascending=False))

# Logistic regression testing example so drop na values
df = df.dropna()
print("Framingham shape without na values: ", df.shape)
print(df.head())

# Create dataframes for random forest classification
X = df.drop(['TenYearCHD'], axis = 1)
print(X.shape)
print(X.head())

y = df['TenYearCHD']
print(y.shape)
print(y.head())

# Create training and test datasets
x_train ,x_test ,y_train ,y_test = train_test_split(X,y,test_size=0.2)

# Scale train and test data
def preprocess_data(X, scaler = None):
    if scaler is None:
        scaler = StandardScaler().fit(X)  # Fit only on training data
    X_scaled = scaler.transform(X)       # Transform using the scaler
    return X_scaled, scaler

X_train_scaled, scaler = preprocess_data(x_train)
X_test_scaled, _ = preprocess_data(x_test, scaler = scaler)


# Convert to DataFrame for easy viewing
X_train_scaled = pd.DataFrame(X_train_scaled, columns = x_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns = x_test.columns)


print(X_train_scaled.head())
print(y.head())

# Create, tune & return random forest using grid cross validation
def create_tune_random_forest(X, y, param_grid = None, cv = 5, scoring='accuracy'):
    
    # Default parameter grid if none provided
    if param_grid is None:
        param_grid = {
            'n_estimators': [10, 20, 30],
            'max_depth': [None, 1, 2],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
}

    # Initialize RandomForestClassifier and GridSearchCV
    rf_model = RandomForestClassifier(class_weight = 'balanced', random_state=42)
    grid_search = GridSearchCV(
        estimator = rf_model,
        param_grid = param_grid,
        cv = cv, 
        scoring = scoring,
        n_jobs = -1,
        verbose = 2
        )

    # Fit GridSearchCV to the data
    grid_search.fit(X, y)

    # Print detailed results for each parameter combination
    for mean_score, params in zip(grid_search.cv_results_['mean_test_score'], grid_search.cv_results_['params']):
        print(f"Mean CV Score: {mean_score:.4f} | Parameters: {params}")

    return grid_search.best_estimator_

best_rf_model = create_tune_random_forest(X_train_scaled, y_train)

# Predictions
y_predict = best_rf_model.predict(X_test_scaled)
y_predict_proba = best_rf_model.predict_proba(X_test_scaled)[:, 1]  # Probability of positive class

# Convert predicted values to dataframe
y_predict = pd.DataFrame(y_predict) # Maybe this change alters class label counts???

# Accuracy
accuracy = accuracy_score(y_test, y_predict)
print('\n')
print(f"Accuracy: {accuracy:.2f}")
print('\n')

# Create confusion matrix for plotting
confusion_m = confusion_matrix(y_test, y_predict, labels = [0, 1])

# Use Seaborn heatmap to plot
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_m, annot = True, fmt = "d", cmap = "Blues", cbar = False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Random Forest Confusion Matrix")
plt.show()

# Classification report - trying to figure out why class label counts are incorrect in Confusion Matrix
print('y_test unique values: ', np.unique(y_test))
print('y_test unique value count: ', y_test.value_counts())
print('\n')

print('y_predict unique values: ', np.unique(y_predict))
print('y_predict unique value count: ', y_predict.value_counts())
print('\n')


report = classification_report(y_test, y_predict)
print("Classification Report:\n", report)
print('\n')

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_predict_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.show()