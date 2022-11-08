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
import sys

%matplotlib auto

np.set_printoptions(threshold=sys.maxsize)

# THIS IS  A GITHUB TEST, HI ELLIE

#%%------- Question 1 ------- 

# Split matrix A into lower triangular L and upper triangular U
# A*x = L*U*x = L*y = b with U*x = y
# Choose Lii = 1

def LU_Decomp(A):
    
    # Initialise L and U matrices:
    
    L = np.zeros([len(A), len(A)])
    U = np.zeros([len(A), len(A)])
    
    for i in range(len(A)):
        L[i][i] = 1
    
    # Perform decomposition:
    
    for i in range(len(A)):
        for j in range(len(A)):
            sumu = 0
            suml = 0
            
            if i<=j:
                for k in range(i):
                    sumu += (L[i][k] * U[k][j])
                U[i][j] = A[i][j] - sumu
            
            if i>=j:
                for l in range(j):
                    suml += (L[i][l] * U[l][j])
                L[i][j] = (1/U[j][j]) * (A[i][j] - suml)
            
    print(pd.DataFrame(L))
    print('')
    print(pd.DataFrame(U))
    return pd.DataFrame(L), pd.DataFrame(U)



A = [[5, 4, 3, 2, 1],
     [4, 8, 6, 4, 2],
     [3, 6, 9, 6, 3],
     [2, 4, 6, 8, 4],
     [1, 2, 3, 4, 5]]

b = [[0],
     [1],
     [2],
     [3],
     [4]]


L, U = (LU_Decomp(A))


def Forward_Sub(L, b):
    
    # Initialise y:
    
    y = np.zeros([len(L[0]), len(b[0])])
    
    y[0] = b[0] / L[0][0]
    
    for i in range(1, len(L)):
        summ = 0
        for j in range(0, i):
            summ += L[j][i] * y[j]
        print(summ)
        y[i] = (b[i] - summ) / L[i][i]
        
    print(pd.DataFrame(y))
    return pd.DataFrame(y)
        
    

y = Forward_Sub(L, b)




def Backward_Sub(U, b):
    
    # Initialise x:
    
    x = np.zeros([len(U[0]), len(b[0])])
    
    N = len(b)
    
    x[N-1] = b[N-1] / U[N-1][N-1]
    # print(x[0])
    
    
    
    for i in range(0, N):
        summ = 0
        for j in range(i+1, N-1):
            summ += U[j][i] * x[j][0]
        # print(summ-b[i][0])
        # print(U[i][i])
        x[i] = int(b[i][0] - summ) / U[i][i]
        
    print(pd.DataFrame(x))
    return pd.DataFrame(x)

x = Backward_Sub(U, b)

#%%------- Question 5 b) ------- 


T = ([[1, 0.526, 0.257, 0, 0, 0],
     [0.526, 1, 0.64, 0, 0, 0],
     [0.257, 0.64, 1, 0, 0, 0],
     [0, 0, 0, -1, -0.581, -0.978],
     [0, 0, 0, -0.581, -1, -0.5],
     [0, 0, 0, -0.978, -0.5, -1]])


# Perform Gauss_Jordan elimination to find inverse T^-1
# Apply matrix manipulation to A * A^-1 = UNIT until you get A^-1 = M * UNIT


def Unit(n):
    # mat = np.zeros((n, n))
    # for i in range(n): 
    #     mat[[i],[i]] = mat[[i],[i]] + 1
    mat = np.diag(np.ones(6))
    return mat
            

#augmat = 
# def MatMult()

#out = np.matmul(T, M1)
# print(out)

#thing = np.zeros((5, 5))
#thing[[0],[0]] = thing[[0],[0]] +1
# print(thing)

print(pd.DataFrame(T))
print(Unit(6))

aug = np.concatenate((T, Unit(6)), axis=1)
print(pd.DataFrame(aug))


def GaussJordan(M):
    aug = np.concatenate((M, Unit(len(M))), axis=1)
    for k in range(len(M)):
        for i  in range(len(M)):
            if k!=i:
                ratio = aug[i][k] / aug[k][k]
                for j in range(len(aug[0])):
                    aug[i][j] -= (ratio * aug[k][j])
        aug[k] = aug[k] / aug[k][k]
    inverse = aug[0:len(M),len(M):]
    return inverse
    # np.savetxt('out.txt', inverse)



# T[0][0]+= T[0][0]-1
# print(pd.DataFrame(T))
Tinverse = GaussJordan(T)

print(pd.DataFrame(Tinverse))



#%%------- Question 5 c) ------- 


# Tinverse = np.array([[1.40, -0.86, 0.19, 0, 0, 0],
#      [-0.86, 2.24, -1.22, 0, 0, 0],
#      [0.19, -1.22, 1.74, 0, 0, 0],
#      [0, 0, 0, -34.71, 3.91, 32.09],
#      [0, 0, 0, 3.91, -1.77, -2.95],
#      [0, 0, 0, 32.09, -2.95, -31.00]], dtype=object)

Tinverse = GaussJordan(T)

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

zero_I = Dot(Tinverse, zero_B)


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















