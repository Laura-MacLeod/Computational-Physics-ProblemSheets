#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 09:03:27 2022

@author: laura
"""

# ------- PROBLEM SHEET 4 --------


import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import scipy as sp


#%% ------- QUESTION 3 a) --------

%matplotlib inline

'''
x' = v
v' = t

grad of x is v, grad of v is t
grad of x at 0 is 1/3

'''

# Function that takes initial values for x(0) and x'(0) / v(0) to return values for all x and v for t
# Only works when second derivative is equal to t (linear)

def EulerIVP(tlower, tupper, N, x0, x0prime):
    x = []
    v = []
    t = np.linspace(tlower, tupper, N)
    trange = tupper - tlower
    Δt = trange/N
    
    # Gradient of v is just t (known)
    # v can be found
    
    vn = x0prime
    
    for i in range(len(t)):
        vn1 = vn + t[i] * Δt
        v.append(vn1)
        vn = vn1
        
    # Gradient of x is v
    # v is now known so x can be found
    
    xn = x0
    
    for i in range(len(t)):
        xn1 = xn + v[i] * Δt
        x.append(xn1)
        xn = xn1
    
    return t, v, x

    

def Analytic(t):
    return (t*(t**2 + 2))/6

def AnalyticDiff(t):
    return t**2/2 + 1/3

x1vals = np.linspace(0, 1, 100)
y1vals = Analytic(x1vals)

x2vals = np.linspace(0, 1, 100)
y2vals = AnalyticDiff(x2vals)


t, v, x = EulerIVP(0, 1, 100, 0, 1/3)


plt.rcParams['font.size'] = 15

plt.figure(1)
plt.plot(x1vals, y1vals)
plt.ylabel('x(t)')
plt.xlabel('t')

plt.figure(1)
plt.plot(t, x)

    
plt.figure(2)
plt.plot(x2vals, y2vals)
plt.ylabel('dx(t)/dt ')
plt.xlabel('t')

plt.figure(2)
plt.plot(t, v)


plt.figure(3)
plt.plot(x1vals, x1vals)
plt.ylabel('d2x(t)/dt2 ')
plt.xlabel('t')












#%% ------- QUESTION 3 b) --------

%matplotlib inline
plt.rcParams['font.size'] = 15

'''
x(0) = 0
x(1) = 1/2

need to guess x'(0) / v(0)
'''

'''
Function accpeting a list/array of initial v guesses, plots v and x values as function of guesses
'''

def EulerBVP_GuessFunction(tlower, tupper, N, x0, x1, guesses):
    

    t = np.linspace(tlower, tupper, N)
    trange = tupper - tlower
    Δt = trange/N
    
    # Use guess for v(0) or xprime(0)
    
    manyvends = []
    xendsfunc = []
    
    for j in guesses:
        
        x = []
        v = []
        
        v0 = j
        
        for i in range(len(t)):
            vn1 = v0 + t[i] * Δt
            v.append(vn1)
            v0 = vn1

            
        xn = x0
        
        for i in range(len(t)):
            xn1 = xn + v[i] * Δt
            x.append(xn1)
            xn = xn1
            
        vend = v[-1:]
        xend = x[-1:]
        
        print(xend)
        
        manyvends.append(vend)
        xendsfunc.append(xend)
    
    xendsfunc = np.array(xendsfunc)
    xendsfunc -= x1
    
    plt.figure(4)
    plt.plot(guesses, manyvends)
    plt.xlabel('v0 guesses')
    plt.ylabel('v end values')

    plt.figure(5)
    plt.plot(guesses, xendsfunc)
    plt.xlabel('v0 guesses')
    plt.ylabel('x end values')
    
    
    return guesses, manyvends, xendsfunc
    

# v0, vend, xend = EulerBVP_GuessFunction(0, 1, 100, 0, 1/2, np.linspace(0, 1, 50))













'''
Function that takes initial values for x(0) and and a guess for x'(0) / v(0) 
Return residual of x between projected x_end and known x1
'''

def EulerBVP(tlower, tupper, N, x0, x1, guess):
    

    t = np.linspace(tlower, tupper, N)
    trange = tupper - tlower
    Δt = trange/N
    
    # Use guess for v(0) or xprime(0)

    x = []
    v = []
    
    v0 = guess
    
    for i in range(len(t)):
        vn1 = v0 + t[i] * Δt
        v.append(vn1)
        v0 = vn1

        
    xn = x0
    
    for i in range(len(t)):
        xn1 = xn + v[i] * Δt
        x.append(xn1)
        xn = xn1
        
    vend = v[-1:]
    xend = x[-1:]
    
    residual = float(xend[0]) - x1
    
    return residual


residual = EulerBVP(0, 1, 100, 0, 1/2, 1/3)



'''
Function that accepts an upper and lower guess for x'(0) / v(0), as well as the EulerBVP function 
and its parameters, exluding the final 'guess' parameter

Returns value for x'(0) / v(0) to given precision and plots a graph to show iteration process
'''

def Bisection(vlguess, vrguess, function, tlower, tupper, N, x0, x1, precision):
    
    initial_vlguess = vlguess
    initial_vrguess = vrguess
    
    xlguess = function(tlower, tupper, N, x0, x1, vlguess)
    xrguess = function(tlower, tupper, N, x0, x1, vrguess)
    
    if xlguess > 0 or xrguess < 0:
        print("Initial guesses do not bracket root, please retry")
        return
    
    error = 1
    
    midv_vals = []
    
    while error > precision:
    
        midv = (vrguess + vlguess)/2
        midv_vals.append(midv)
        
        xrguess = function(tlower, tupper, N, x0, x1, vrguess)
        xlguess = function(tlower, tupper, N, x0, x1, vlguess)
        
        error = xrguess - xlguess
        
        midx = function(tlower, tupper, N, x0, x1, midv)
        
        if midx > 0:
            vrguess = midv
        elif midx < 0:
            vlguess = midv
    
    plt.plot(midv_vals)
    plt.xlabel('Number of iterations')
    plt.ylabel('Value of v(0)')
    plt.title('Lower guess: %.0f, Upper guess: %.0f' % (initial_vlguess, initial_vrguess))
    
    print('v0 or xprime0 is found to be %.10f with an error of %.10f' % (midv_vals[-1:][0], error))

    return midv_vals[-1:]
    
    
Bisection(-100, 100, EulerBVP, 0, 1, 100, 0, 1/2, 0.001)













