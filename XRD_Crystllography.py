# -*- coding: utf-8 -*-
"""
Created on Fri May 24 16:22:16 2024

@author: Hugh Littlehailes
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import scipy
import seaborn as sns
import os
import csv
from glob import glob
import math

a = 2.94e-10
b = 2.94e-10
c = 4.64e-10
Lambda = 1.540598e-10

def hexagonal(h, k, l):
    d2 = 1/(((4/3)*((h**2)+(k**2)+(h*k)) + (l**2)*(a**2/c**2))*(1/(a**2)))
    d = np.sqrt(d2)
    theta1 = np.arcsin(Lambda/(2*d))
    theta = theta1*(360/np.pi)
    return theta


peak1 = hexagonal(1, -1, 0 )

print(peak1)

            