# -*- coding: utf-8 -*-
"""
Created on Sun Sep 27 04:46:01 2020

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

total_trials = np.arange(0,100)
error_vector = np.zeros(100)
gradient_norm_vector = np.zeros(100)
iteration_vector = np.zeros(100)
trial = 0
lam = 0.01
u = 0.042
n = 0.6
for trial in total_trials:
    print('Trial : \n', trial)
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
    w = w_b[0:30]
    b = w_b[30]
    s_f = np.zeros(31)
    s = s_f[0:30]
    f = s_f[30]
    delta_w_b = np.zeros(31)
    delta_s_f = np.zeros(31)
    
    N = np.arange(0,500)
    filler = np.arange(0,30)
    
    inf = 0
    iter = 0
    
    while inf == 0:
        iter = iter + 1
        i=0
        j=0
        for j in filler:
            sum1 = 0
            i=0
            for i in N:
                sum1 = sum1 + (-y_train[i])*(x_train[i][j]) + (((math.exp(np.dot(w,x_train[i,:])+b))/(1+math.exp(np.dot(w,x_train[i,:])+b)))*(x_train[i][j]))    
            factor_sum1 = sum1 + (lam)*(w[j])
            
            delta_w_b[j] = factor_sum1
            
        i = 0
        j = 0
            
        b_fac1 = (np.sum(y_train))*(-1)
        
        sum_bfac2 = 0
        for i in N:
            sum_bfac2 = sum_bfac2 + ((math.exp(np.dot(w,x_train[i,:])+b))/(1+math.exp(np.dot(w,x_train[i,:])+b)))
        
        delta_w_b[30] = b_fac1 + sum_bfac2
        
        i = 0
        
        r=0
        t=0
        for r in filler:
            sum1_z = 0
            t = 0
            for t in N:
                sum1_z = sum1_z + (-y_train[t])*(x_train[t][r]) + (((math.exp(np.dot(s,x_train[t,:])+f))/(1+math.exp(np.dot(s,x_train[t,:])+f)))*(x_train[t][r])) 
            factor_sum1_z = sum1_z + (lam)*(s[r])
            
            delta_s_f[r] = factor_sum1_z
            
        r=0
        t=0
        
        f_fac1 = b_fac1
        
        sum_ffac2 = 0
        for t in N:
            sum_ffac2 = sum_ffac2 + ((math.exp(np.dot(s,x_train[t,:])+f))/(1+math.exp(np.dot(s,x_train[t,:])+f))) 
        
        delta_s_f[30] = f_fac1 + sum_ffac2

        t=0
        sum_func = 0
        
        for l in N:
            sum_func = sum_func + (-y_train[l])*((np.dot(w,x_train[l,:]))+b) + math.log(1+(math.exp((np.dot(w,x_train[l,:]))+b)))
            
        function_value = sum_func + (0.5)*(lam)*((np.linalg.norm(w))**2)
        
        norm_delta = np.linalg.norm(delta_w_b)
        norm_delta_squared = norm_delta*norm_delta
        
        if norm_delta_squared <= (10**(-6))*(1+abs(function_value)):
            break
        else:
            if iter == 1:
                w_b_old = w_b
                w_b = w_b - (u)*(delta_w_b)
                s_f = w_b + n*(w_b - w_b_old)
            else:
                w_b_old = w_b
                w_b = s_f - (u)*(delta_s_f)
                s_f = w_b + n*(w_b - w_b_old)
    
    
    iteration_vector[trial] = iter
    gradient_norm_vector[trial] = norm_delta 
    
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
    
    error_vector[trial] = error_percent


print('The vector of errors in all the trials is obtained : \n',pd.DataFrame(error_vector, columns = ['Error'], index = range(1,101)))
print('The vector of gradient norms to check appropriate convergence is : \n',pd.DataFrame(gradient_norm_vector, columns = ['Gradient Norms'], index = range(1,101)))
print('The vector of iterator counts in each trial obtained is : \n',pd.DataFrame(iteration_vector,columns = ['iterations'],index = range(1,101)))

average_error = np.mean(error_vector)
average_iterations = np.mean(iteration_vector)

print('The average iterations till the accuracy of 10^-6 is satisfied in all 100 trials is : \n',average_iterations)
print('The average error obtained in all the 100 trials with 500 iterations in each trial is : \n',average_error)
print('The average error in percentage is : \n', average_error*100)
print('The average error in terms of average test data output mismatches is : \n', average_error*69)


    
    
                
                                              