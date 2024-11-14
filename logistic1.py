#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 22:18:54 2024

@author: frankbogle
"""
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
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

# Create range of plots displaying categorical independent varibale & target
education_chd_counts = df.groupby(['education', 'TenYearCHD']).size().unstack()

# Plot with 'education' on x-axis and separate bars for each 'TenYearCHD' category
education_chd_counts.plot(kind = 'bar', xlabel = "Education", ylabel = "Counts", title = "Counts of TenYearCHD by Education Level", width = 0.8)
plt.xticks(rotation = 45)
plt.show()

# Create dataframes for logistic regression
X = df.drop(['TenYearCHD'], axis = 1)
print(X.shape)
print(X.head())

y = df['TenYearCHD']
print(y.shape)
print(y.head())

# Create training and test datasets
x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

# Scale train and test data
def preprocess_data(X, scaler = None):
    if scaler is None:
        scaler = StandardScaler().fit(X)  # Calculate mean & std
    X_scaled = scaler.transform(X)       # Scale data
    return X_scaled, scaler

X_train_scaled, scaler = preprocess_data(x_train)
X_test_scaled, _ = preprocess_data(x_test, scaler=scaler)

# Create train & cross validate model
def train_logistic_cv(X, y, cv = 5):
    # Create pipeline with scaling and cross-validated logistic regression
    pipeline = make_pipeline(
        StandardScaler(),
        LogisticRegressionCV(class_weight = 'balanced', cv=cv, scoring='accuracy', max_iter = 1000, random_state = 42)
    )
    
    # Train the model
    pipeline.fit(X_train_scaled, y_train)
    
    # Return the trained pipeline
    return pipeline

# Call the function and get the trained model
trained_model = train_logistic_cv(X_train_scaled, y_train)

# Use trained model for predictions or evaluations
y_predict = trained_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_predict)
print(f"Accuracy: {accuracy:.2f}")

# Create confusion matrix for plotting
confusion_m = confusion_matrix(y_test, y_predict)

# Use Seaborn heatmap to plot
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_m, annot = True, fmt = "d", cmap = "Blues", cbar = False)
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.title("Logistic Regression Confusion Matrix")
plt.show()

# Classification report
report = classification_report(y_test, y_predict)
print("Classification Report:\n", report)








