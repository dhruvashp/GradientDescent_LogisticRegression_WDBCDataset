# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 06:51:29 2020

@author: DHRUV
"""




"""
HW1
d
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

term_vector = np.zeros(500)
x_train_i,x_test_i,y_train_i,y_test_i = train_test_split(features,output,train_size = 500)
x_train_i.index = range(1,501)
x_test_i.index = range(1,70)
y_train_i.index = range(1,501)
y_test_i.index = range(1,70)

x_train = x_train_i.to_numpy()
x_test = x_test_i.to_numpy()
y_train = y_train_i.to_numpy()
y_test = y_test_i.to_numpy()


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
grad_fill = np.arange(0,30)
i = np.arange(0,500)
epoch = 0
for epoch in i:
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
    
    norm_grad_squared = (norm_grad)*(norm_grad)
    
    r=0
    sum_function = 0
    for r in N:
        sum_function = sum_function + ((-y_train[r])*(np.dot(w,x_train[r,:])+b)) + math.log(1 + math.exp((np.dot(w,x_train[r,:]))+b))
    
    function_value = sum_function + 0.5*lam*((np.linalg.norm(w))**2)
    term_vector[epoch]=(norm_grad_squared/(1+abs(function_value)))
    w_b = w_b - (u)*(grad_w_b)


w = w_b[0:30]        
b = w_b[30]


y_test_predicted = np.zeros(69)
test_index = np.arange(0,69)
alpha = 0
for alpha in test_index:
    probab_dot = np.dot(w,x_test[alpha])
    probab_expo = probab_dot + b
    exponential = math.exp((probab_expo)*(-1))
    probab = 1/(1+exponential)
    if probab >= 0.5:
        y_test_predicted[alpha] = 1
    else:
        y_test_predicted[alpha] = 0

beta = 0
error_sum = 0
for beta in test_index:
    if y_test[beta] != y_test_predicted[beta]:
        error_sum = error_sum + 1

error_percent = error_sum / 69

print(term_vector)


plt.plot(i[1:500],term_vector[1:500],'r', marker=2)
plt.xlabel('t')
plt.ylabel('For Gradient Descent Ratio')


print('Error Percent is : \n', error_percent)


