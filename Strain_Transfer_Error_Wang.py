# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 16:09:02 2024

@author: Hugh Littlehailes
Strain transfer error from Wang H. & J.-G. Dai Composites Part B 162 (2019) 303–313 
"""
import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.integrate import quad
import scipy.special as special
from numpy import sqrt, sin, cos, pi
import sympy as sm
#x=sm.Symbol('x')

Ef = 7e10      #Young's modulus of the fibre
rf = 1.25e-4   #Radius of the fibre
Gp = 2.0e7     #Shear modulus of the fibre protective layer
rp = 1.95e-4   #Radius of the fibre protective layer
Ga = 1.24e8    #Shear modulus of the adhesive
ha = 0.5e-4   #Thickness of adhesive layer between substrate and fibre
#L=np.arange(1e-4,4e-3, 1e-4)
L1 = np.arange(1e-4,1.0001e-2, 1e-4)
L2 = np.arange(1e-4,8.001e-3, 1e-4)
L3 = np.arange(1e-4,5.0001e-3, 1e-4)
L4 = np.arange(1e-4,2.0001e-3, 1e-4)
alpha_f = 6.5e-6 #µm/m K-1
alpha_m = 11.7e-6 #µm/m K-1
##L = 5e-3


''''Wang's values'''
#Ef = 7.2e10
#rf = 1.25e-4
#Gp = 6.65e7
#rp = 4e-3
#Ga = 2.26e8
#ha = 2e-3
#L1 = np.arange(1e-3,4.0001e-2, 1e-3)
#L2 = np.arange(1e-3,2.5001e-2, 1e-3)
#L3 = np.arange(1e-3,2.0001e-2, 1e-3)
#L4 = np.arange(1e-3,1.0001e-2, 1e-3)
#alpha_f = 0.55e-6 #µm/m K-1
#alpha_m = 23.2e-6 #µm/m K-1
#L = 5e-3
#x = Symbol('x')
#def integrand()

A = (rp)/(Ef*pi*(rf)**2)
C = (ha+rp/Ga)
D = (rp/Gp)*np.log(rp/rf)
E = rp/Ga

def integrand(x):
    #return (1/(C-(E*sin(x))+D))
    #return (1/((ha+rp-rp*sin(x))/Ga)+(rp/Gp)*np.log(rp/rf))
    return (1/(((ha+rp-rp*np.sin(x))/Ga)+(rp/Gp)*np.log(rp/rf)))
#def integration(C, rp, Ga, D, x):
#    return 

B, error = quad(integrand, 0, pi)
#print(B)

beta_a1 =[]
beta_a2 =[]
beta_a3 =[]
beta_a4 =[]

Lambda = sqrt(float(A)*float(B))
for j in L1:
#B = integration(C, E, D)
    
    LL1 = Lambda*L1
    LL2 = Lambda*L2
    LL3 = Lambda*L3
    LL4 = Lambda*L4
    #Eps_f
    beta_a1.append(1-((np.sinh(LL1))/(LL1*np.cosh(LL1))) + (alpha_f/alpha_m)*(np.sinh(LL1)/(LL1*np.cosh(LL1))))
    beta_a2.append(1-((np.sinh(LL2))/(LL2*np.cosh(LL2))) + (alpha_f/alpha_m)*(np.sinh(LL2)/(LL2*np.cosh(LL2))))
    beta_a3.append(1-((np.sinh(LL3))/(LL3*np.cosh(LL3))) + (alpha_f/alpha_m)*(np.sinh(LL3)/(LL3*np.cosh(LL3))))
    beta_a4.append(1-((np.sinh(LL4))/(LL4*np.cosh(LL4))) + (alpha_f/alpha_m)*(np.sinh(LL4)/(LL4*np.cosh(LL4))))


print(beta_a1[0])
L1 = L1[::-1]*100
L2 = L2[::-1]*100
L3 = L3[::-1]*100
L4 = L4[::-1]*100
#print(len(L1))
fig, ax = plt.subplots( constrained_layout = True)
ax.scatter(L1, beta_a1[0], label = '4 cm')
ax.scatter(L2, beta_a2[0], label = '2.5 cm')
ax.scatter(L3, beta_a3[0], label = '2 cm')
ax.scatter(L4, beta_a4[0], label = '1 cm')
ax.legend()
ax.set(xlabel = "Half the bonding length L(cm)")#, ylabel = r'Strain transfer coefficient [$\Beta$]')
plt.ylabel(r'Strain transfer coefficient $\beta$') 
ax.set_ylim([-0.01,1])
#ax.set_xticks([Time[0],Time[-1]])
#ax.set_xticklabels([time_axis_min, time_axis_max],rotation = 0, fontsize=9)
plt.show()

#print(beta_a)