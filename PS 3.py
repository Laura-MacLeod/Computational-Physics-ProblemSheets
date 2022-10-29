#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 08:59:53 2022

@author: laura
"""

# ------- PROBLEM SHEET 3 --------


import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd


#%% ------- QUESTION 1 --------

%matplotlib inline

# Define Gaussain function

def Gaussian(x, σ, μ):
    out = 1 / (σ * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - μ)/σ)**2)
    return out

# Plot Gaussian

x = np.linspace(-4, 4, 4)

plt.scatter(x, Gaussian(x, 1, 0))


# Create Lagrange Polynomial function

def Lagrange(n, x, f):
    P = []
    for k in range(0, n):
        m=k
        Pn = 0
        for i in range(m):
            
            temp = f[i]
            for j in range(m):
                if j != i:
                    temp = temp * ((x - x[j]) / (x[i] - x[j]))
            Pn +=(temp)
            
        P.append(Pn)
        
    print(P)
    return P


def Poly(P, x):
    lis = []
    for j in range(len(x)):
        out = 0
        for i in range(len(P)):
            out += (P[i] * x[j]**i)
            print(out)
        lis.append(out)
    return lis




P = Lagrange(4, x, Gaussian(x, 1, 0))

x_new = np.linspace(-4, 4, 100)

lagran = Poly(P, x_new)
print('START')
print(lagran)

plt.plot(x_new, lagran)

# plt.plot(x, Pn)






# find P0:
#     P = []
#     for k in range(0, n):
#         m=n
#         Pn = 0
#         while m>=0:
#             for i in range(0, m):
#                 inner=1
#                 for j in range(0, m):
#                     compute fraction
#                     inner * fraction
#                 temp = inner * f[n]
#                 Pn +=(temp)
#                 m -= 1
    
    
    








#%% --------- QUESTION 4 ------------

# GAUSS ELIMINATION


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib
import sys

%matplotlib auto

np.set_printoptions(threshold=sys.maxsize)

#%%------- Question 5 b) ------- 


T = ([[1, 0.526, 0.257, 0, 0, 0],
     [0.526, 1, 0.64, 0, 0, 0],
     [0.257, 0.64, 1, 0, 0, 0],
     [0, 0, 0, -1, -0.581, -0.978],
     [0, 0, 0, -0.581, -1, -0.5],
     [0, 0, 0, -0.978, -0.5, -1]])


# Perform Gauss elimination to find inverse x
# Apply matrix manipulation to A * x = b until you get x = M * b


def Gauss(A, b):
    aug = np.concatenate((A,b), axis=1)
    for k in range(len(A)):
        for i  in range(len(A)):
            if k!=i:
                ratio = aug[i][k] / aug[k][k]
                for j in range(len(aug[0])):
                    aug[i][j] -= (ratio * aug[k][j])
        aug[k] = aug[k] / aug[k][k]
    inverse = aug[:, len(A):]
    return inverse
    # np.savetxt('out.txt', inverse)


A = [[1, 1, 1, 1],
     [2, 1, -1, -2],
     [2, 0.5, 0.5, 2],
     [4/3, 1/6, -1/6, -4/3]]

b = [[0], [1], [0], [0]]

x = Gauss(A, b)

print(pd.DataFrame(x))







#%% ------- QUESTION 5 a) --------

%matplotlib inline
np.seterr(divide='ignore', invalid='ignore')

def Trapezium(f, x):
    xupper = x[-1:]
    xlower = x[0]
    n = len(f)
    rang = xupper - xlower
    h = rang / n
    temp = 0
    
    for i in range(0, n):
        if np.isnan(f[i]) == True:
            continue
        if i==0 or i==(n-1):
            temp += (0.5 * f[i])
        else:
            temp += f[i]
    out = h * temp
    print('The area of this function is: %0.4f, found using %i trapeziums and a stepsize of %0.4f' % ((out[0]), n-1, h))
    return out

def Gaussian(x, σ, μ):
    return (1 / (σ * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - μ)/σ)**2))


# Find area of Gaussian

x = np.linspace(-4, 4, 500)
y = Gaussian(x, 1, 0)
plt.scatter(x, y)
#area = Trapezium(y, x)




def SineFunc(x):
    return (np.sin(x**2) / x)

x = np.linspace(0, 10, 1000000)
y = SineFunc(x)

Trapezium(y, x)


#%% ------- QUESTION 6 a) --------


%matplotlib inline


def DFT(f, t): # input arrays of y and x coordinates
    N = len(f) # number of samples, must be even
    T = t[-1:] - t[0] # domain length
    Δt = T / N # sample spacing
    ωmin = (2 * np.pi) / T  # also = Δω
    ωmax = (2 * np.pi) / (2 * Δt)
    p = np.arange(-N/2, (N/2)+1, 1) # integer index
    # ωp = p * ωmin
    n = np.arange(0, N, 1)
    # tn = n * Δt
    
    f_tilda = []
    
    for i in p:
        ωp = i * ωmin
        fp_tilda = 0
        for j in n:
            fn = f[j]
            tn = t[j]
            fp_tilda += fn * np.exp(complex(0, ωp * tn))
            
        f_tilda.append(abs(fp_tilda))
    
    ωp = (p * ωmin)
    return ωp, f_tilda
    




# ------ TRANSFORMING PROVIDED ALIGNMENT OF MOLECULE DATA -------

data = np.loadtxt("DFT_data")

xdata=[]
ydata=[]

for i in range(len(data)):
    xdata.append(data[i][0])
    ydata.append(data[i][1])


# Plot data in time domain

plt.figure(3)
plt.plot(xdata, ydata)
plt.xlabel('t(ps)')
plt.ylabel('f(t)')


# Plot data in frequency domain

plt.figure(4)
ω, FT = DFT(ydata, xdata)
plt.scatter(ω, FT, marker='x')
plt.xlim(0, 8)








# NOT MY FUNCTION, TAKEN FROM INTERNET
def gaussian(x, alpha, r):
          return 1/(np.sqrt(alpha**np.pi))*np.exp(-alpha*(x - r)**2)

    
# ------ TESTING WITH GAUSSIAN FUNCTION ------
    
plt.figure(1)
t = np.linspace(0, 2*np.pi, 151)
gaus = gaussian(t, 500e-15, 0) + gaussian(t, 500e-15, 2*np.pi)
# sint = np.sin(t*5)
plt.plot(t, gaus, 'o')

plt.figure(2)
# ω = (2 * np.pi) / t
ωp , gaus_tilda = DFT(gaus, t)
plt.plot(ωp, gaus_tilda, 'o')


 










