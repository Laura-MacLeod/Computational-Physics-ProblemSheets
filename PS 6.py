#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 16 10:10:24 2022

@author: laura
"""

# ------- PROBLEM SHEET 6 --------


import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
import scipy as sp


#%% ------- QUESTION 5 a) (i) --------

%matplotlib inline

x = np.linspace(0, 1, 100)
t = np.linspace(0, 1, 80)

rho = [list(0.5 * (1 + np.sin(2*np.pi*x)))]

Δt = t[1] - t[0]
h = x[1] - x[0]
v = 1

a = Δt * v / h




for i in range(len(t)):
    rho.append([])
    
    for j in range(len(x)-1):
    
        pin = rho[i][j]
        
        if j == 0:
            pi1n = rho[i][(len(x))-2]
        else:
            pi1n = rho[i][j-1]
        
        pin1 = (1 - a) * pin + a * pi1n
        
        rho[i+1].append(pin1)
    
    plt.figure(i)
    plt.plot(x[:-1], rho[i+1])
    plt.xlabel('x')
    num = i/100
    plt.ylabel('time %0.2d' % num)


'''
To avoid periodic discontinuity, only find p_i^n+1 up to x-1 because boundary conditions mean that
x[0]=x[I].
Also cut off end of x.

Final state is perfect sinusoid.
'''





#%% ------- QUESTION 5 a) (ii) --------

%matplotlib inline

def Advection(t0, tn, tN, x0, xn, xN, initial_rho, left_rho, right_dpdx):
    
    x = np.linspace(x0, xn, xN)
    t = np.linspace(t0, tn, tN)
    
    rho = [initial_rho]                         # Initial conditions (now ~independent of final state)
    #rho = [list(0.5 * (1 + np.sin(2*np.pi*x)))]
    
    Δt = t[1] - t[0]
    h = x[1] - x[0]
    v = 1
    
    a = Δt * v / h
    
    # pint = np.arcsin(2*t - 1) / (2*np.pi) 

    ptx0 = left_rho                             # Dirichlet BC for p(t, 0) = p_in (t)
    
    dpdx1 = right_dpdx                         # Neumann BC for dp/dx (t, 1)
    
    
    for i in range(len(t)):
        rho.append([ptx0[i]])                   # Add left-hand BC
    
    for i in range(len(t)):
        for j in range(1, len(x)-1):            # -2 to make room for boundary conditions
        
            pin = rho[i][j]
            pi1n = rho[i][j-1]
            
            pin1 = (1 - a) * pin + a * pi1n
            
            rho[i+1].append(pin1)
        
        pin = rho[i][len(x)-1]
        neumann_bc = pin + Δt * v * dpdx1[i]    # Add right-hand BC
        rho[i+1].append(neumann_bc)
        
        plt.figure(i)
        plt.plot(x, rho[i+1])
        plt.xlabel('x')
        num = i/100
        plt.ylabel('time %0.2d' % num)


initial_rho = np.zeros(len(x))
left_rho = 0.5 * (1 + np.sin(2*np.pi*t))
right_dpdx = np.zeros(len(t))

Advection(0, 1, 150, 0, 1, 100, initial_rho, left_rho, right_dpdx)


'''
Final state is negative sinusoid with high gradient drop to zero at x=1 due to initial condiitons
and Neumann BC.
'''


#%% ------- QUESTION 5 b) (i) --------

%matplotlib auto

# time, ρ, T, vx, by, bz = pd.read_csv('Wind_Data.csv', skiprows=1)
columns = ['time', 'rho', 'T', 'vx', 'by', 'bz']
vals = pd.read_csv('Wind_Data.csv', encoding="ascii", usecols=columns)

time = vals['time']
ρ = vals['rho']
T = vals['T']
vx = vals['vx']
by = vals['by']
bz = vals['bz']


t = np.linspace(0, 1, 150)
R_L1 = 1.6e6 # [km]

time = list(time)
# print(time.index(60))


def Linear_Interpolate(x, xi, yi):
    y = []
    x = list(x)

    for i in range(0, len(x)):
        
        upper_diffs = []
        upper_xs = []
        upper_ys = []
        for j in range(len(xi)):
            val = xi[j] - x[i]
            if val >= 0:
                upper_diffs.append(val)
                upper_xs.append(xi[j])
                upper_ys.append(yi[j])
        
        min_diff = min(upper_diffs)
        x1index = upper_diffs.index(min_diff)
        xiplus1 = upper_xs[x1index]
        yiplus1 = upper_ys[x1index]
        
        
        lower_diffs = []
        lower_xs = []
        lower_ys = []
        for j in range(len(xi)):
            val = x[i] - xi[j]
            # print(val)
            if val >= 0:
                lower_diffs.append(val)
                lower_xs.append(xi[j])
                lower_ys.append(yi[j])
        
        min_diff = min(lower_diffs)
        x0index = lower_diffs.index(min_diff)
        xiplus0 = lower_xs[x0index]
        yiplus0 = lower_ys[x0index]
   
        
        val = ((xiplus1 - x[i])*yiplus0 + (x[i] - xiplus0)*yiplus1) / (xiplus1 - xiplus0)
            
        y.append(val)
        
    return(y)





t = np.linspace(0, time[-1], 1000)
x = np.linspace(-R_L1, 0, 1441)

f = Linear_Interpolate(t, time, ρ)
# print(f)

plt.figure(1)
plt.plot(x, ρ)

# interpolated_ρ = Linear_Interpolate(t, time, ρ)

# plt.plot(t, interpolated_ρ, color='orange')




# x1 = [0, 3, 4, 4.5, 7, 8, 10]

# y1 = [2, 3, 5, 4, 7, 3, 5]

# plt.figure(2)
# plt.plot(x1, y1, 'o', ls='-')

# x2 = np.linspace(0, 10, 250)

# y2 = Linear_Interpolate(x2, x1, y1)

# plt.plot(x2, y2)



initial_rho = []
left_rho = 0.5 * (1 + np.sin(2*np.pi*t))
right_dpdx = np.zeros(len(t))

Advection(0, time[-1], 1000, -R_L1, 0, 100, initial_rho, left_rho, right_dpdx)










