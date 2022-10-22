#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 16 23:21:06 2022

@author: laura
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib

%matplotlib auto

#%%------- Question 5 b) ------- INCOMPLETE


T = [[1, 0.526, 0.257, 0, 0, 0],
     [0.526, 1, 0.64, 0, 0, 0],
     [0.257, 0.64, 1, 0, 0, 0],
     [0, 0, 0, -1, -0.581, -0.978],
     [0, 0, 0, -0.581, -1, -0.5],
     [0, 0, 0, -0,978, -0.5, -1]]


# Perform Gauss_Jordan elimination to find inverse T^-1
# Apply matrix manipulation to A * A^-1 = UNIT until you get A^-1 = M * UNIT


def Unit(n):
    mat = np.zeros((n, n))
    for i in range(n): 
        mat[[i],[i]] = mat[[i],[i]] + 1
    return mat
            
unit = Unit(6)


# def MatMult()

out = np.matmul(T, M1)
# print(out)





thing = np.zeros((5, 5))
thing[[0],[0]] = thing[[0],[0]] +1
# print(thing)





Tinverse = np.array([[1.40, -0.86, 0.19, 0, 0, 0],
     [-0.86, 2.24, -1.22, 0, 0, 0],
     [0.19, -1.22, 1.74, 0, 0, 0],
     [0, 0, 0, -34.71, 3.91, 32.09],
     [0, 0, 0, 3.91, -1.77, -2.95],
     [0, 0, 0, 32.09, -2.95, -31.00]], dtype=object)




#%%------- Question 5 c) ------- 


Tinverse = np.array([[1.40, -0.86, 0.19, 0, 0, 0],
     [-0.86, 2.24, -1.22, 0, 0, 0],
     [0.19, -1.22, 1.74, 0, 0, 0],
     [0, 0, 0, -34.71, 3.91, 32.09],
     [0, 0, 0, 3.91, -1.77, -2.95],
     [0, 0, 0, 32.09, -2.95, -31.00]], dtype=object)

HER = pd.read_csv('Data/HER_data')
HBK = pd.read_csv('Data/HBK_data')
TSU = pd.read_csv('Data/TSU_data')
KMH = pd.read_csv('Data/KMH_data')

zero_Bx1 = HER['Bx'][0]
zero_By1 = HER['By'][0]

zero_Bx2 = HBK['Bx'][0]
zero_By2 = HBK['By'][0]

zero_Bx3 = TSU['Bx'][0]
zero_By3 = TSU['By'][0]


zero_B = [[zero_Bx1], [zero_Bx2], [zero_Bx3], [zero_By1], [zero_By2], [zero_By3]]
print(zero_B)

# Create dot product function

def Dot(m1, m2):
    out = np.zeros((len(m1), len(m2[0])))
    for n in range(len(m1[0])): #2
        for j in range(len(m2)): #2
            for m in range(len(m1)): #2
                for i in range(len(m2[0])): # 1
                    if n==j:
                        out[m][i] += m1[m][n] * m2[j][i]
    # print(out)
    return out


# ---- Use I = T^-1 * B -----

# zero_I = Dot(Tinverse, zero_B)


# ----- Use B' = T' * I ------

T_prime = [[0.552, 0.998, 0.61, 0, 0, 0],
           [0, 0, 0, -0.988, -0.52, -0.998]]

zero_B_prime = Dot(T_prime, zero_I)

#%%


# ----- Apply to all timesteps ------

# print(HER)

Bx1 = HER['Bx']
By1 = HER['By']

Bx2 = HBK['Bx']
By2 = HBK['By']

Bx3 = TSU['Bx']
By3 = TSU['By']

KMH_B_x = KMH['Bx']
KMH_B_y = KMH['By']

temp_B = [[Bx1], [Bx2], [Bx3], [By1], [By2], [By3]]
B = []
time = []

print(HER.iloc[0][0])

# print(list(HER.iloc[1]))

for i in range(0, len(HER)):
    B.append(list(([HER.iloc[i][1]], [HBK.iloc[i][1]], [TSU.iloc[i][1]], [HER.iloc[i][2]], [HBK.iloc[i][2]], [TSU.iloc[i][2]])))
    time.append(HER.iloc[i][0])

B = np.array([B])
time = np.array(time)
time = pd.to_datetime(time)

zero_I = Dot(Tinverse, zero_B)


B_x = []
B_y = []

for i in range(len(B[0])):
    I = Dot(Tinverse, B[0][i])
    B_prim = Dot(T_prime, I)
    B_x.append(B_prim[0])
    B_y.append(B_prim[1])

    
# for i in range(len(B[0])):
#     B_x.append(B_prime[i][0])
#     B_y.append(B_prime[i][1])

dates = matplotlib.dates.date2num(time)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(time, B_x, color='blue', label='Interpolated')
plt.plot(time, KMH_B_x, color='orange', label='Measured')
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.ylabel('$B_x [nT]$ ')
plt.xlabel('Date-Time')

plt.subplot(1, 2, 2)
plt.plot(time, B_y, label='Interpolated')
plt.plot(time, KMH_B_y, color='orange', label='Measured')
plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.ylabel('$B_x [nT]$ ')
plt.xlabel('Date-Time')
plt.legend()


#%%















