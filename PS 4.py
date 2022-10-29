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


#%% ------- QUESTION 1 --------


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



























