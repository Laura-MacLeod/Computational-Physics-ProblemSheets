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



def LinCong(n, a, b, c, x0):
    randoms = []
    x = x0
    for i in range(n):
        x = (a * x + c) % b
        randoms.append(x)
    print(randoms)
    return randoms

# LinCong(15, 7, 11, 1)

'''
Only 10 unique numbers can be generated due to the use of the 
With a=7, b=12 and x0=1, the numbers are just 7s and 1s, which are obviously not random
Can't seem to fix this by changing a and x0 individually, but changing both of them seems to fix it
'''

LinCong(20, 3, 5, 1, 0)










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


MonteCarlo(Func, 0, 10, 100)





#%% ------- QUESTION 5 --------


# Find s

def S_Iteration(T, H, s0):
    s = s0
    for i in range(10000):
        # while abs(s-temp) < 0.0000000000000000001:
            temp = s
            s = np.tanh((1/T)*(H/1 + s))
            # print(s)
            if abs(s-temp) < 1e-10:
                return s
                break

# S_Iteration(0.5, 0)


# Question (i): Generate plot for M vs T (M = s)

Tvals = list(np.linspace(1e-10, 1.2, 100))
svals = []

for i in Tvals:
    svals.append(S_Iteration(i, 0, 1))

plt.figure(1)
plt.plot(Tvals, svals)
plt.ylabel("Mean Magnetisarion M")
plt.xlabel("T (T_c/T, so T=1 at critical T)")



# Question (ii): Generate plot for net energy E vs T (E = s*(H/Hc + s))

Evals = []

def E(T):
    s = S_Iteration(T, 0, 1)
    return -s*(0/1 + s)

for i in Tvals:
    val = E(i)
    Evals.append(val)

plt.figure(2)
plt.plot(Tvals, Evals)
plt.ylabel("Energy E")
plt.xlabel("T  (1/T, so T=1 at critical T)")


# Question (iii): Generate plot for net energy C vs T (C = dE/dT)

# Function for derivative using finite difference scheme

def FDS(x, func, n, y0):
    rang = x[-1:] -  x[0]
    h = rang/n
    val = y0
    
    plot = []
    
    for j in range(len(x)):
        temp = val
        val = (func(x[j]+h) - func(x[j])) / h
        # print(val)
        plot.append(val)
    return(plot)


C = FDS(Tvals, E, 100, 1)

plt.figure(3)
plt.plot(Tvals, C)
plt.ylabel("Heat Capacity C")
plt.xlabel("T  (1/T)")



# Question (b)

'''
Only the mean magnetism changes when the spin is changed to s0=-1. This is becuase the polarity
cancels out in the energy, and and heat capacity is derived from the energy. The direction of spin
has no effect on the energy, but it does change the direction the polarisation approaches zero from, 
because the spins are becoming perfectly disordered from opposite polarities.
'''


# Question (c)

# Skip for now, move onto Monte Carlo methods



lattice = np.array([[1, -1, -1, -1, 1], 
                   [1, -1, -1, 1, -1],
                   [1, 1, 1, 1, -1],
                   [1, -1, 1, -1, 1],
                   [1, 1, -1, 1, -1]])


# ------- Function to multiply every matrix element's value by the values of its nearest neightbours, then sum

def NearestNeighbours(matrix): # n is matrix height, m is matrix width
    n = len(matrix)
    m = len(matrix[0])
    out = 0
    for i in range(n):
        for j in range(m):
            for k in range(n):
                for l in range(m):
                    si = matrix[i][j]
                    sj = matrix[k][l]
                    if ((k==i+1 or k==i-1) and (l==j)) or ((k==i) and (l==(j+1) or l==(j-1))):
                        # print(si, sj)
                        val = si * sj
                        out += val
    return(out)

# NearestNeighbours(lattice)


# ------ Function to find the tolal energy of a lattice

def TotalEnergy(matrix, J, H):
    spin_sum = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            spin_sum += matrix[i][j]
    E = -J * NearestNeighbours(matrix) - H * spin_sum
    return E

# print(TotalEnergy(lattice, 1, 0))
    

# ------- Function to flip spins until minimum energy state is reached

def Flipping(matrix, T):
    energy = TotalEnergy(lattice, 1, 0)
    tag = 0
    spin_sum = 0
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            spin_sum += matrix[i][j]
            initial_spin = matrix[i][j]
            if initial_spin == 1:
                matrix[i][j] = -1
            elif initial_spin == -1:
                matrix[i][j] = 1
                
            new_energy = TotalEnergy(lattice, 1, 0)
            change_energy = new_energy - energy
            boltz = np.exp(-change_energy / T)
            
            if change_energy < 0:
                energy = new_energy
                # print('FLIP, ΔE < 0')
                tag += 1
                continue
            elif change_energy > 0:
                if np.random.rand() <= boltz:
                    energy = new_energy
                    # print('FLIP, ΔE > 0')
                    tag += 1
                    continue
                else:
                    matrix[i][j] = initial_spin
            
            # print(matrix, tag)
    spin_mean = abs(spin_sum) / (len(matrix) * len(matrix[0]))
    return matrix, tag, spin_mean
            # print(matrix)
                    
# final_lattice, tag = Flipping(lattice, 0.999999)


# Plot the mean spin of a lattice after repeated flipping iterations, until the energy is minimised

def Average_Spin(matrix, T, n):
    final_lattice = matrix
    tag=1
    counter = 0
    iterations = []
    spin =  []
    
    # while tag > 0:
    while counter < n:
        final_lattice, tag, spin_mean = Flipping(final_lattice, T)
        spin.append(spin_mean)
        counter += 1
        iterations.append(counter)
    
    plt.figure(4)
    # plt.plot(iterations, spin)
    plt.xlabel("Number of iterations")
    plt.ylabel("Magnitude of Mean Spin")
    plt.scatter(iterations[-1:], spin[-1:], color='red')
    
    ave_spin = np.mean(spin)
    
    return ave_spin



# Repeat the energy minimisation to retrieve values of mean spin 

# ave_spin = Average_Spin(lattice, 3, 300)

ave_spin = []

T = np.linspace(1e-10, 10, 10)

for i in T:
    val = Average_Spin(lattice, i, 300)
    ave_spin.append(val)

# Plotting Mean Magnetisation

plt.figure(5)
plt.plot(T, ave_spin)
plt.xlabel("Temperature T")
plt.ylabel("Mean Magnetisation M (Monte Carlo)")

# Plotting Energy

EvalsMC = []
for i in ave_spin:
    val = -i*(0/1 + i)
    EvalsMC.append(val)

plt.figure(6)
plt.plot(T, EvalsMC)
plt.xlabel("Temperature T")
plt.ylabel("Energy E (Monte Carlo)")

# Plotting Heat Capacity



'''
Figure 5 and 6 are generated using the Monte Carlo Method.
Earlier, figure 1 and 2 represent the same relationship but generated using the Mean Field Approximation.

It can be seen that the Monte Carlo method changes randomly between iterations, since it suffers from
random temperature fluctuations, so spins can randomly flip to increasing energy.

With the Mean Field Approximation, the mean spin is at zero for all temperatures above the critical
temperature. With the Monte Carlo technique, it takes much higher temperatures, above 10Tc, for the 
spin to average to zero. This is due to the temperature fluctuations.
'''








