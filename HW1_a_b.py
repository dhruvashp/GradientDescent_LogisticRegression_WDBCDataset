# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 23:32:03 2020

@author: DHRUV
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math as math

df=pd.read_csv('wdbc.data', header = None)
print(df)

"""
1. (a) and (b)
removing patient ID (column 0)

"""

df=df.drop(columns = 0)
print(df)
df.columns = ['O',1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
print(df)

"""
renaming column header
Header O is for Output of y
Header 1 to 30 are for our 30 features

"""

"""
Now, normalizarion
"""

df_features = df.drop(columns = 'O')
print(df_features)

features = df_features.to_numpy()
print(features)
print(features.shape)
c = np.zeros((1,30))
range = np.arange(0,569)
i=0
for i in range:   
    c = c + features[i]

print(c)
mean = c / 569
print(mean)
j=0
for j in range:
    features[j] = features[j] - mean

print(features)

t=0
for t in range:
    features[t]=features[t] / np.linalg.norm(features[t])

print(features)

df_new_features = pd.DataFrame(data = features)
print(df_new_features)
df_new_features['O']=df['O']
print(df_new_features)

df_new_features.to_csv('Normalized_Dataset.csv')

"""
The new .csv file contains normalized feature vector in the first 30 rows with the 
output vector in the final row. This normalized file will be used for the next
parts of the assignment

"""

    
    
