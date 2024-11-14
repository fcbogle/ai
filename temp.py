# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
print("Hello Frank")

import numpy as np
import pandas as dp
print(np.__version__)
df = dp.read_csv("/Users/frankbogle/Downloads/titanic.csv")
print(df.to_string())

missing_values = df.isna()
print(missing_values)


