# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 13:53:53 2024

@author: Hugh Littlehailes
"""

import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import quad
import scipy.special as special
from numpy import sqrt, sin, cos, pi
import sympy as sm

Ef = 7e10
rf = 1.25e-4
Gp = 2.0e7
rp = 1.95e-4
Ga = 1.24e8
ha = 0.5e-4
#L=np.arange(1e-4,4e-3, 1e-4)
L1 = np.arange(1e-4,1.0001e-2, 1e-4)
L2 = np.arange(1e-4,8.001e-3, 1e-4)
L3 = np.arange(1e-4,5.0001e-3, 1e-4)
L4 = np.arange(1e-4,2.0001e-3, 1e-4)
alpha_f = 6.5e-6 #µm/m K-1
alpha_m = 11.7e-6 #µm/m K-1





R = 1=(cosh(kx))/(cosh(kL)) + (phi_f*del_T)/()
