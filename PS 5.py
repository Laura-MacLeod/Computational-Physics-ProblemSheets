#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  9 09:03:27 2022

@author: laura
"""

# ------- PROBLEM SHEET 5 --------


import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import scipy as sp


#%% ------- QUESTION 2 a) --------


def EulerHarmonic(tlower, tupper, N, u0, v0):
    
    t = np.linspace(tlower, tupper, N)
    trange = tupper - tlower
    Δt = trange / N
    
    u = [u0]
    v = [v0]
    energy = [u0**2 + v0**2]
    
    un = u0
    vn = v0
    
    for i in range(N-1):
        un1 = un + vn * Δt
        vn1 = vn - un * Δt
        
        u.append(un1)
        v.append(vn1)
    
        e = un1**2 + vn1**2
        energy.append(e)
        
        un = un1
        vn = vn1
    
    plt.figure(1)
    plt.plot(t, u, label='Numeric')
    plt.xlabel('t')
    plt.ylabel('u')
    
    plt.figure(2)
    plt.plot(t, energy)
    plt.xlabel('t')
    plt.ylabel('energy')
    plt.ylim(0, 2)
    
    # plt.figure(3)
    # plt.plot(t, v)
    # plt.xlabel('t')
    # plt.ylabel('v')
    
    
    return t, u, energy


def HarmonicOscillator(tlower, tupper, N):
    t = np.linspace(tlower, tupper, N)
    plt.figure(1)
    plt.plot(t, np.sin(t), label='Analytic')
    plt.legend()
    return np.sin(t)


EulerHarmonic(0, 1, 10, 0, 1)
HarmonicOscillator(0, 1, 100)

'''
Needs minimum of 2 points to run. The fewer the number of points / larger the step size, the worse
the approximation.
'''


#%% ------- QUESTION 2 b) --------





def Iteration(tlower, tupper, N, un, vn, uguess, vguess):

    un1 = uguess
    vn1 = vguess
    
    t = np.linspace(tlower, tupper, N)
    trange = tupper - tlower
    Δt = trange / N
    
    for i in range(10000):
        # while abs(s-temp) < 0.0000000000000000001:
            tempu = un1
            tempv = vn1
            un1 = un + Δt * (vn + Δt * un1)
            vn1 = vn + Δt * (un + Δt * vn1)
            # print(s)
            if abs(un1-tempu) < 1e-10 and abs(vn1-tempv) < 1e-10:
                print(un1, vn1)
                return un1, vn1
                break

Iteration(0, 1, 100, 0, 1, 0.1, 0.9)



def EulerHarmonicImplicit(tlower, tupper, N, u0, v0, uguess, vguess):
    
    t = np.linspace(tlower, tupper, N)
    trange = tupper - tlower
    Δt = trange / N
    
    u = [u0]
    v = [v0]
    energy = [u0**2 + v0**2]
    
    un = u0
    vn = v0
    
    for i in range(N-1):
        un1, vn1 = Iteration(tlower, tupper, N, un, vn, uguess, vguess)
        
        u.append(un1)
        v.append(vn1)
    
        e = un1**2 + vn1**2
        energy.append(e)
        
        un = un1
        vn = vn1
    
    plt.figure(1)
    plt.plot(t, u, label='Numeric')
    plt.xlabel('t')
    plt.ylabel('u')
    
    plt.figure(2)
    plt.plot(t, energy)
    plt.xlabel('t')
    plt.ylabel('energy')
    plt.ylim(0, 2)
    
    # plt.figure(3)
    # plt.plot(t, v)
    # plt.xlabel('t')
    # plt.ylabel('v')
    
    
    return t, u, energy


EulerHarmonicImplicit(0, 1, 100, 0, 1, 0.1, 0.9)
HarmonicOscillator(0, 1, 100)





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








#%% ------- QUESTION 3 b) --------

'''

'''





























