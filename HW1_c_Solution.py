# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 12:40:56 2020

@author: DHRUV
"""


"""
HW1_c_Part2

Actual Solution
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

total_trials = np.arange(0,100)
error_vector = np.zeros(100)
gradient_norm_vector = np.zeros(100)
trial = 0
for trial in total_trials:
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
    
    w = w_b[0:30]        
    b = w_b[30]
    gradient_norm_vector[trial] = norm_grad
    
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
    
    error_vector[trial] = error_percent


print('The vector of errors in all the trials is obtained : \n',pd.DataFrame(error_vector, columns = ['Error'], index = range(1,101)))
print('The vector of gradient norms to check appropriate convergence is : \n',pd.DataFrame(gradient_norm_vector, columns = ['Gradient Norms'], index = range(1,101)))

average_error = np.mean(error_vector)

print('The average error obtained in all the 100 trials with 500 iterations in each trial is : \n',average_error)
print('The average error in percentage is : \n', average_error*100)
print('The average error in terms of average test data (total 69 points) output mismatches is : \n', average_error*69)


"""

This program will take some time to run. For me it was about 5-10 minutes

The error vector collects the error in per unit for each trial 

Thus the error vectors shows net error in all trials according to mismatches

The gradient norm vector collects the gradient norms at convergence (after iterations are complete
or after the norm becomes <= 0.001)

The gradient norm vector shows converge occurs approximately as they are almost zero (though not quite)

The gradient norm vector has norms in the range of 0.5-0.6

If the iterations would be increased beyond 500 we would surely see them down to almost 0
as was elucidated in Part1 of the c portion of the HW

Again the calculation of w and b in w_b is done similar to part 1. The function
was partially differentiated by hand with respect to w's and b and the equations so obtained
were then represented via the calculations above

The loops for iterations are same as that in Part 1 and the only thing different
here is the addition of trials

The average value of error was calculated from the error vector

The basic strategy for gradient descent was thus -

Obtain the gradient equations for w and b. One term in each was independent of the external
iterations and thus was calculated in a vector for w, grad_w_1 and a scalar for b grad_b_1

The term dependent of w and b was then calculated as the ratio_vector which was filled
at the beginning of each iteration with the latest w and b values

Then the loop calculated the actual gradient values, after which the next w_b estimate was 
made

Finally p was calculated. Using p we made the decision whether y=1 or y=0. This y was then
placed in a vector of y_test_predicted which was initially declared as 0. 

Mismatches were obtained by comparing these y to the actual test y

Average error in mismatches is error in per unit multiplied by the total data length over 
which error was evaluated, here 69 for the total test data length

"""

    
            
    
    