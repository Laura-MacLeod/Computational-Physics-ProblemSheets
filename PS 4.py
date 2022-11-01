#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 08:59:53 2022

@author: laura
"""

# ------- PROBLEM SHEET 4 --------


import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import scipy as sp


#%% ------- QUESTION 1 --------


=======
def LinCong(n, a, b, x0):
    randoms = []
    x = x0
    for i in range(n):
        x = (a * x) % b
        randoms.append(x)
    print(randoms)
    return randoms

LinCong(15, 7, 11, 1)

'''
Only 10 unique numbers can be generated due to the use of the 
With a=7, b=12 and x0=1, the numbers are just 7s and 1s, which are obviously not random
Can't seem to fix this by changing a and x0 individually, but changing both of them seems to fix it
'''












#%% ------- QUESTION 3 ATTEMPT 1 --------

# Attempt 1

%matplotlib inline

def UnitGaussian(x):
    out = 1 / (0.5 * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - 1)/0.5)**2)
    return out



def MonteCarlo(func, lower, upper, N):
    # lower = x[0]
    # upper = x[-1:]
    
    rands = []
    
    for i in range(N):
        rands.append(sp.random.uniform(lower, upper))
    rands = np.array(rands)
    
    fx = func(rands)
    
    plt.figure(1)
    plt.scatter(rands, fx)
    
    top_f = max(fx)
    rang = upper - lower
    V = top_f * rang
    
    I = 0
    
    for i in range(1, N):
        I += fx[i]
    
    I = I * (V/N) 
    print(I)
    return I



MonteCarlo(UnitGaussian, -1, 3, 100000)



#%% ------- QUESTION 3 ATTEMPT 2 --------


# Attempt 2

%matplotlib inline

def UnitGaussian(x):
    out = 1 / (1 * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - 1)/1)**2)
    return out



def MonteCarlo(func, xlower, xupper, N):
    # xlower = x[0]
    # xupper = x[-1:]
    
    xrands = []
    
    for i in range(N):
        xrands.append(sp.random.uniform(xlower, xupper))
    xrands = np.array(xrands)
    
    fx = func(xrands)

    # plt.scatter(rands, fx)
    
    fupper = max(fx)
    flower = min(fx)
    
    frands = []
    
    for i in range(N):
        frands.append(sp.random.uniform(flower, fupper))
    frands = np.array(frands)
    
    xrang = xupper - xlower
    frang = fupper - flower
    V = frang * xrang
    
    
    I = 0
    fscatters = []
    xscatters = []
    
    for i in range(1, N):
        if frands[i] <= fx[i]:
            I += frands[i]
            fscatters.append(frands[i])
            xscatters.append(xrands[i])
    
    # I = I * (V/N) 
    I = len(fscatters) * V /N
    plt.scatter(xscatters, fscatters)
    x = np.linspace(xlower, xupper, 1000)
    plt.plot(x, func(x))
    print(I)
    return I



# MonteCarlo(UnitGaussian, -3, 5, 10000)




# Trying for the function 

def Func(x):
    return np.sin(x**2)/x


MonteCarlo(Func, 0, 10, 10000)






