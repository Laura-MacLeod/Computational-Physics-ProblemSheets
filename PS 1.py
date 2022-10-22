#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 09:35:26 2022

@author: laura
"""

'''
Question 4 - -calculate 1/x numerically

Iteration 

Create separate Taylor function for 1/(1+g)
Add f(i+1) and f(i) and you should get 1 exactly.
Must have g<1

'''

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

%matplotlib inline

#%%


# ------- Question 1 b) -------



byte1 = int('0000110', 2)

byte2 = int('00010001', 2)

byte3 = format(17, 'b')

print(byte1)
print(byte2)
print(byte3)



#%%


# ------- Question 2 c) -------


print(1 + 0.000000000000001)



#%%

# ------- Question 4 -------


def taylor(g, n):
    value = 1
    for j in range(1, n):
        value += (-1)**j * g**j
    return value


# print(1/(1+0.6))
# print(taylor(0.6, 100))


def iteration(g):
    n=1
    diff = 1
    while abs(diff)>10**-6:
        n+=1
        # diff = taylor(g, (n+1)) - taylor(g, n)
        diff = 1/(1+g) - taylor(g, n)
        if n>20**3: #---- breaks loop after ~ 40 seconds
            break
    print("Exact:", 1/(1+g))
    print("(i+1) Term:", taylor(g, n+1))
    print("(i) Term:", taylor(g, n))
    print("Difference:", diff)
    print("Number of Iterations:", n)
    return g, 1/(1+g), taylor(g, n+1), taylor(g, n), diff, n
    # print("f(i+1) + g*f(i):", (taylor(g, n+1) + g * taylor(g, n)))

iteration(-0.6)


lis = []

for i in np.arange(-0.9, 1, 0.1):
    # lis.append(iteration(i))
    a = pd.DataFrame([iteration(i)], columns=['g', 'Exact', 'iplus1', 'i', 'Difference', 'Iterations'])
    lis.append(a)
    # print(iteration(i))


# print(lis)


df = pd.concat(lis)
print(df)
plt.figure(1)
plt.xlabel('x')
plt.ylabel('1/x')
plt.scatter(df.g +1, df.Exact)
plt.scatter(df.g +1, df.iplus1)
plt.scatter(df.g +1, df.i)

plt.figure(2)
plt.xlabel('x')
plt.ylabel('1/x Differences')
plt.plot(df.g +1, abs((df.Exact - df.i)/df.Exact))

plt.figure(3)
plt.xlabel('x')
plt.ylabel('Iterations n')
plt.scatter(df.g +1, df.Iterations, marker='o')



# iplus1 - exact    / exact

# part d) see iterations graph.

