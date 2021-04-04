# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 10:12:40 2020

@author: DHRUV
"""
"""
HW1_c_Part1

Gradient Descent Logic Check

"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import math as math
from sklearn.model_selection import train_test_split

df=pd.read_csv('Normalized_Dataset.csv', index_col=0)
print(df)
df.columns = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,'O']
print(df)
df.index = range(1,570)
print(df)
features = df.drop(columns = 'O')
print(features)
output = df['O']
print(output)

x_train_i = features.iloc[range(0,500)]
print(x_train_i)
x_test_i = features.iloc[range(500,569)]
print(x_test_i)
x_test_i.index = range(1,70)
print(x_test_i)
y_train_i = output.iloc[range(0,500)]
print(y_train_i)
y_test_i = output.iloc[range(500,569)]
print(y_test_i)
y_test_i.index = range(1,70)
print(y_test_i)


x_train = x_train_i.to_numpy()
x_test = x_test_i.to_numpy()
y_train = y_train_i.to_numpy()
y_test = y_test_i.to_numpy()



print(x_train)
print(x_test)
print(y_train)
print(y_test)

w_b = np.zeros(31)
grad_w_b = np.zeros(31)
ratio_vector = np.zeros(500)
lam = 0.01
u = 0.042


grad_w_1 = np.zeros(30)

grad_w_1_index = np.arange(0,30)

N = np.arange(0,500)


k=0
for k in grad_w_1_index:
    q=0
    sum = 0
    for q in N:
        sum = sum + (-y_train[q])*(x_train[q][k])
    grad_w_1[k] = sum
    
k=0
q=0

grad_b_1 = (np.sum(y_train))*(-1)


iterations = 500

iter = np.arange(0,iterations)
grad_fill = np.arange(0,30)

i = 0

for i in iter:
    w = w_b[0:30]
    b = w_b[30]
    l=0
    for l in N:
        dot = np.dot(w,x_train[l,:])
        expo = dot + b
        raised = math.exp(expo)
        ratio_vector[l] = (raised)/(raised+1)
    j=0
    for j in grad_fill:
        sum_raise = 0
        d=0
        for d in N:
            sum_raise = sum_raise + (ratio_vector[d])*(x_train[d][j])
        grad_w_b[j] = sum_raise + grad_w_1[j] + (lam)*(w_b[j])
        
    grad_w_b[30] = grad_b_1 + np.sum(ratio_vector)
    
    norm_grad = np.linalg.norm(grad_w_b)
    
    if norm_grad <= 0.001:
        break
    else:
        w_b = w_b - (u)*(grad_w_b)


print(w_b)
print(grad_w_b)
print(norm_grad)


"""

This part 1 of c was only to check my gradient descent algorithm and logic on a 
definite model known. Here train data was the first 500 rows and the test data
was the last 60 rows. The gradient equation was first obtained via hand calculations
and those equations obtained were then modelled here

Again only the gradient descent logic was checked. In part 2 the actual problem will
be solved

Also Gradient Norm for 500 iterations was obtained 0.60

When I increased iterations, I was able to get a Norm almost 0 (0.0009)

"""
    
    
    
    
    




