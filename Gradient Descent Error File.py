# -*- coding: utf-8 -*-
"""
Created on Thu Sep 24 02:08:19 2020

@author: DHRUV
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

"""
labels for features and output changed according to general 
matrix norms

In the first part to solving c we'll run only a single trial
on the first 500 rows of the data set as training data set and the
remaining 69 rows of the data set as test data

This is done to get acquainted with the entire modelling process
of gradient descent in Python

"""

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

"""
The first 30 entries represent w's and the last entry b
This initial w and b is taken to be 0 in entirety
"""

u = 0.2

index = np.arange(0,500)
grad_cal_index = np.arange(0,30)
N_index = np.arange(0,500)

lam = 0.01

"""

i iterates
k is to evaluate entries of gradient, the entries corresponding
to partial derivatives with respect to w's

for partial derivative with respect to b, the equation changes and so 
the last entry in grad_w_b will be seperately calculated

q moves over x_train to evaluate the gradient

The equation for gradient was obtained by partially differentiating the
given base equation in HW1 by hand. The formulas thus obtained for the 
gradient are then represented here


"""


i=0
for i in index:
    w = w_b[0:30]
    b = w_b[30]
    k=0   
    for k in grad_cal_index:
        q=0
        sum_1 = 0
        sum_2 = 0
        for q in N_index:
            sum_1 = sum_1 + (-y_train[q])*(x_train[q][k])
            exponent_interim = np.dot(w,x_train[q,:])
            exponent = exponent_interim + b
            e_exponent = math.exp(exponent)
            ratio = (e_exponent)/(e_exponent + 1)
            sum_term = (ratio)*(x_train[q][k])
            sum_2 = sum_2 + sum_term
        
        grad_w_b[k] = sum_1 + sum_2 + (lam)*(w_b[k])
         
    
    t=0
    sum_1_b = 0
    sum_2_b = 0
    for t in N_index:
        sum_1_b = sum_1_b + (-y_train[t])
        exponent_interim_b = np.dot(w,x_train[t,:])
        exponent_b = exponent_interim_b + b
        ratio_b = (exponent_b)/(exponent_b + 1)
        sum_2_b = sum_2_b + ratio_b
    
    grad_w_b[30] = sum_1_b + sum_2_b
    
    norm_grad = np.linalg.norm(grad_w_b)
    
    if norm_grad <= 0.001:
        break
    else:
        w_b = w_b - u*grad_w_b
    

print(grad_w_b)
print(norm_grad)
print(w_b)





"""
A few comments

This is not the most efficient code, even though accurate. Reasons are thus -

sum_1_b term is constant for each iteration and thus can be calculated and
stored in advance

sum_1 can be calculated in advance and stored as a vector. Not dependent on descent
iteration. When gradient for w are calculated, their corresponding 
indices can be called

w'x + b is different for each iteration but they can be calculated
in the beginning of each iteration for different rows and placed in 
a vector and called on in the q loop rather than being calculated repeatedly
for each q loop in each k loop

Again we presume that the goal here isn't to write the most efficient
code, in so far as the code is correct, which this code is

"""

   


              
         
           
         
          
         