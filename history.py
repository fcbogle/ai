# -*- coding: utf-8 -*-
# *** Spyder Python Console History Log ***

## ---(Sun Nov  3 21:53:16 2024)---
python --version
python -V
python --version
clear
%runcell -i 0 /Users/frankbogle/.spyder-py3/temp.py
pwd
ls
ls -al
cd Downloads/
ls -al
pwd
ls
ls *.csv
pwd
%runfile /Users/frankbogle/.spyder-py3/temp.py --wdir
norows df
len(df)
df.shape
sum(df.isnull())
sum(df.isna())
df.isnull()
df.isnull().sum()
clear
df.describe()
df.isna()
df.columns
df.columns.count()
df.types
df.dtypes
df.info(verbose=true)
df.info(verbose=True)
df.select_dtypes(include='bool')
df.values
df.head()
pip --version
pip
python --version
pip install schedule

## ---(Mon Nov  4 08:22:11 2024)---
pip --version

## ---(Mon Nov  4 08:24:34 2024)---
python --version
conda --version
pip --version
runfile('/Users/frankbogle/.spyder-py3/temp.py', wdir='/Users/frankbogle/.spyder-py3')
runcell(0, '/Users/frankbogle/.spyder-py3/temp.py')
print(np.__version__)

## ---(Mon Nov  4 08:42:24 2024)---
pip --verrsion
pip --version

## ---(Mon Nov  4 08:44:45 2024)---
import sys
sys.executable

## ---(Mon Nov  4 08:47:40 2024)---
pip --version
conda --version
python --version
%runfile /Users/frankbogle/.spyder-py3/temp.py --wdir
import sys
print(sys.version)
df = dp.read_csv("/Users/frankbogle/Downloads/titanic.csv")
print(df.to_string())
import numpy as np
import pandas as dp
print(np.__version__)
df = dp.read_csv("/Users/frankbogle/Downloads/titanic.csv")
print(df.to_string())

%runfile /Users/frankbogle/.spyder-py3/test1.py --wdir
%runfile /Users/frankbogle/.spyder-py3/temp.py --wdir
pwd
cd ..
ls
cd Downloads
ls -al
ls -al *.csv
%runfile /Users/frankbogle/.spyder-py3/regression1.py --wdir
print(df.DESCR)
print(df.head())
df['MEDV'] = df.target
df.info()
pwd
df.isnull().sum()
%runfile /Users/frankbogle/.spyder-py3/regression1.py --wdir
print(X_rooms)
print(X_rooms.shape)
print(y_price.shape)
%runfile /Users/frankbogle/.spyder-py3/regression1.py --wdir
print)X_train.shape)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
%runfile /Users/frankbogle/.spyder-py3/regression1.py --wdir
model
model.intercept_
print(model.intercept_, model.coef, model.score(X_train, Y_train))
%runfile /Users/frankbogle/.spyder-py3/regression1.py --wdir
df2 = df.dropna()
X_independent = df2[['RM', 'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'TAX', 'PTRATIO']]
y_dependent = df2['MEDV']
selector = SelectKBest(score_func = f_regression, k=3)
X_train_selected = selector.fit_transform(X_independent, y_dependent)
selected_feature_names = X_train.columns[selector.get_support(indices=True)]
%runfile /Users/frankbogle/.spyder-py3/regression1.py --wdir
df.feature_names
df.head
df.get_feature_names
df.columns[0
]
%runfile /Users/frankbogle/.spyder-py3/regression1.py --wdir
df.columns[9
]
df.columns
%runcell -i 0 /Users/frankbogle/.spyder-py3/regression1.py
%runfile /Users/frankbogle/.spyder-py3/regression2.py --wdir
print(df.shape)
%runfile /Users/frankbogle/.spyder-py3/regression2.py --wdir
df = df.dropna
%runfile /Users/frankbogle/.spyder-py3/regression2.py --wdir
model.predict(y_test)
model.predict(X_test)
model.predict(X_test_selected)
print(model.predict(X_test_selected).shape
()

;
print(model.predict(X_test_selected).shap
z
x

()

;
print(model.predict(X_test_selected))
print(model.predict(X_test_selected)).shaoe
print(model.predict(X_test_selected)).shape
print(y_test)
print(model.predict(X_test_selected)).len()
%runfile /Users/frankbogle/.spyder-py3/regression2.py --wdir
model.predict(X_test_selected)
%runfile /Users/frankbogle/.spyder-py3/regression2.py --wdir
model.predict(X_test_selected)
%runfile /Users/frankbogle/.spyder-py3/regression2.py --wdir
print(y_test)
pwd
cd ../Downloads/
ls -al *.csv
more heart_disease.csv
%runfile /Users/frankbogle/.spyder-py3/regression2.py --wdir
%runfile /Users/frankbogle/.spyder-py3/regression2.py --wdir
print("R^2 Score: ", model.score(X_test_selected, y_test.sorted()))
print("R^2 Score: ", model.score(X_test_selected, y_test)).sorted()
print("R^2 Score: ", model.score(X_test_selected, y_test))
%runfile /Users/frankbogle/.spyder-py3/regression2.py --wdir
print("R^2 Score: ", sorted(model.score(X_test_selected, y_test)), reverse = True)
arr = model.predict(X_test_selected)
redictions = model.predict(X_test_selected)
sorted_predictions = sorted(predictions.flatten(), reverse = True)

predictions = model.predict(X_test_selected)
sorted_predictions = sorted(predictions.flatten(), reverse = True)
print("Predictions in descending order:", sorted_predictions)
%runfile /Users/frankbogle/.spyder-py3/regression2.py --wdir
print("Sorted Selected Features:", sorted_selected_features)
print("Selected Features:", selected_features)
%runfile /Users/frankbogle/.spyder-py3/regression2.py --wdir
pwd
cd ../Downloads/
ls -al f*.csv
%runfile /Users/frankbogle/.spyder-py3/logistic1.py --wdir
print(df.columns)
df.info()
sum(df.isna())
sum(df.isna)
df.isan
fd.isna
df.isna
df.isnull()
%runfile /Users/frankbogle/.spyder-py3/logistic1.py --wdir
%runfile /Users/frankbogle/.spyder-py3/logistic1.py --wdir
pd.crosstab(df['glucose'], df['TenYearCHD']).plot(kind='bar', title="Glucose vs TenYearCHD")
plt.crosstab(df['glucose'], df['TenYearCHD']).plot(kind='bar', title="Glucose vs TenYearCHD")
df.plot(x = "Coronary Risk", y = ['Education'], kind = 'bar')
df.plot(x = "TenYearCHD", y = ['Education'], kind = 'bar')
df.plot(x = "TenYearCHD", y = ['education'], kind = 'bar')
clear
%runfile /Users/frankbogle/.spyder-py3/logistic1.py --wdir
print(y.shape)
print(y.head())
%runcell -i 0 /Users/frankbogle/.spyder-py3/logistic1.py
len(x_train)
len(x_test)
clear
%runfile /Users/frankbogle/.spyder-py3/logistic1.py --wdir
clear
%runfile /Users/frankbogle/.spyder-py3/logistic1.py --wdir
model.score()
model.score(x_test_scaled, y_test)
clear
%runfile /Users/frankbogle/.spyder-py3/logistic1.py --wdir
model.score(x_test_scaled, y_test)
?model.score
?scaler.transform
?scaler.fit_transform
x_train_scaled
x_test_scaled
x_test_scale.shape
x_test_scale.shape()
x_test_scaled.shape()
x_test_scaled.shape
x_train_scaled.shape
model_predict = model.predict(x_test_scaled)
modelpredict
model_predict
y_test
%runfile /Users/frankbogle/.spyder-py3/logistic1.py --wdir
print(confusion_m)
%runfile /Users/frankbogle/.spyder-py3/logistic1.py --wdir
confusion_m
df.head()
model_accuracy = accuracy_score(y_test, model_predict)
clear
%runfile /Users/frankbogle/.spyder-py3/logistic1.py --wdir
accuracy_score
print(model_accuracy)
%runfile /Users/frankbogle/.spyder-py3/logistic1.py --wdir
clear
%runfile /Users/frankbogle/.spyder-py3/logistic1.py --wdir
trained_model = train_and_evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test)
clear
%runfile /Users/frankbogle/.spyder-py3/logistic1.py --wdir
clear
%runfile /Users/frankbogle/.spyder-py3/logistic1.py --wdir
clear
%runfile /Users/frankbogle/.spyder-py3/logistic1.py --wdir
clear
%runfile /Users/frankbogle/.spyder-py3/logistic1.py --wdir
clear
%runfile /Users/frankbogle/.spyder-py3/randfor1.py --wdir
clear
%runfile /Users/frankbogle/.spyder-py3/randfor1.py --wdir
clear
%runfile /Users/frankbogle/.spyder-py3/randfor1.py --wdir
%runfile /Users/frankbogle/.spyder-py3/logistic1.py --wdir
%runfile /Users/frankbogle/.spyder-py3/randfor1.py --wdir
clear
%runfile /Users/frankbogle/.spyder-py3/logistic1.py --wdir
clear
%runfile /Users/frankbogle/.spyder-py3/randfor1.py --wdir
clear
%runfile /Users/frankbogle/.spyder-py3/randfor1.py --wdir
df.hear()
df .head()
df.column()
df.columns()
df.shape()
X.ahape()
X.shape()
%runfile /Users/frankbogle/.spyder-py3/randfor1.py --wdir
print(X.shape())
df = pd.read_csv("/Users/frankbogle/Downloads/framingham.csv")
X = df.drop(['TenYearCHD'], axis = 1)
print(X.shape)
df = df.dropna()
print(df.head())
print(X.shape)
print(df.isna().sum().sort_values(ascending=False))
clear
%runfile /Users/frankbogle/.spyder-py3/randfor1.py --wdir

## ---(Sat Nov  9 13:26:25 2024)---
%debugcell -i 0 /Users/frankbogle/.spyder-py3/randfor1.py

## ---(Sat Nov  9 13:30:49 2024)---
pwd
%runfile /Users/frankbogle/.spyder-py3/randfor1.py --wdir
%runfile /Users/frankbogle/.spyder-py3/logistic1.py --wdir
%runfile /Users/frankbogle/.spyder-py3/randfor1.py --wdir
clear
%runfile /Users/frankbogle/.spyder-py3/logistic1.py --wdir
%runfile /Users/frankbogle/.spyder-py3/randfor1.py --wdir
?df.drop
%runfile /Users/frankbogle/.spyder-py3/logistic1.py --wdir
%runfile /Users/frankbogle/.spyder-py3/randfor1.py --wdir
clear
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

%runfile /Users/frankbogle/.spyder-py3/logistic1.py --wdir
?StandardScaler
>standardscaler.fit
?StandardScaler.fit
?StandardScalar.transform
?StandardScaler.transform
y_test.len()
len(y_test)
clear
%runfile /Users/frankbogle/.spyder-py3/randfor1.py --wdir
pip --version
y_test.unique()
y_test.value_counts()
y_predict.value_counts()
%runfile /Users/frankbogle/.spyder-py3/randfor1.py --wdir
y_test.value_counts()
y_predict.value_counts()
y_predict_series = pd.Series(y_predict)
print(y_predict_series.value_counts())
clear
%runfile /Users/frankbogle/.spyder-py3/randfor1.py --wdir
print(y_predict_series.value_counts())
y_predict.value_counts()
y_predict_series = pd.Series(y_predict)
print(y_predict_series.value_counts())
y_test.value_counts()
?sns.heatmap
%runfile /Users/frankbogle/.spyder-py3/randfor1.py --wdir
clear
%runfile /Users/frankbogle/.spyder-py3/randfor1.py --wdir
y_test.value_counts()
y_predict.value_counts()
y_predict_series = pd.Series(y_predict)
y_predict.value_counts()
y_predict_series = pd.Series(y_predict)
print(y_predict_series.value_counts())
y_test.value_counts()
%runfile /Users/frankbogle/.spyder-py3/logistic1.py --wdir
clear
%runfile /Users/frankbogle/.spyder-py3/logistic1.py --wdir
y_test.value_counts()
y_predict.value_counts()
?confusion_matrix
clear
%runfile /Users/frankbogle/.spyder-py3/randfor1.py --wdir
y_test.value_counts()
y_predict_series = pd.Series(y_predict)
print(y_predict_series.value_counts())
y_train.value_counts()
y_test.value_counts()
y_predict = best_rf_model.predict(X_test_scaled)
y_predict
y_test
clear
%runfile /Users/frankbogle/.spyder-py3/randfor1.py --wdir
np.unique(y_predict)
%runfile /Users/frankbogle/.spyder-py3/randfor1.py --wdir
clear
%runfile /Users/frankbogle/.spyder-py3/randfor1.py --wdir
clear
%runfile /Users/frankbogle/.spyder-py3/randfor1.py --wdir
conda --version
python --version
pwd
ls -al *.py
cd ..
ls
ls -al
pwd
cd
ls
ls -al
pwd
ls -al *.py
ls
pwd
cd .spyder-py3/
ls -al