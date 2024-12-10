# -*- coding: utf-8 -*-
"""
Created on Thu May 30 09:47:00 2024

@author: Hugh Littlehailes
"""

import numpy as np
import csv
from glob import glob
import matplotlib.pyplot as plt
import math
import time
from datetime import datetime


Stripe_5 = {'Zone1Series1': {'x5' :  79441338342572000,
                                 'x4' : -327727475539339000,
                                 'x3' :  540803022112212000,
                                 'x2' : -446205928990579000,
                                 'x1' :  184077849992821000,
                                 'c'  : -30375797316597400},
                'Zone1Series2': {'x5' : -351170127269200000,
                                 'x4' : 1448753774091350000,
                                 'x3' : -2390735612342500000,
                                 'x2' : 1972597550507710000,
                                 'x1' : -813795710439526000,
                                 'c'  :  134292654477756000},
                'Zone2Series1': {'x5' : -577047762716,
                                 'x4' : 2398129429832.21,
                                 'x3' : -3986491891691.48,
                                 'x2' : 3313420605125.79,
                                 'x1' : -1376986444264.45,
                                 'c'  :  228897029384.68},
                'Zone2Series2': {'x5' : 1978396178256,
                                 'x4' : -8222093111247.45,
                                 'x3' : 13668120812779.2,
                                 'x2' : -11360632991644.6,
                                 'x1' : 4721321254529.48,
                                 'c'  :  -784841322219.83}
                }

step1 = (0.825682754722118-0.824629629719839)/9001
#zone1_time = np.arange(0.824629629719839,0.825682754722118,step1)
#zone2_time = np.arange(0.825682870462859,0.837823958452106,104900)
file1 = 'G:/My Drive/LSBU_Trip_Mar2024/Stripe5_Zone1.txt'
file2 = 'G:/My Drive/LSBU_Trip_Mar2024/Stripe5_Zone2_2.txt'
file3 = 'G:/My Drive/LSBU_Trip_Mar2024/Stripe5_data_Complete.txt'

zone1_time = []
zone2_time = []
pow1 = []
pow2 = []
with open(file3, 'r') as file:
    for _ in range(1):
        next(file)
    for line in file:
        columns = line.strip().split('\t')
        zone2_time.append(float(columns[0]))
        #pow1.append(float(columns[1]))
        pow2.append(float(columns[2]))
        
def time_to_seconds(t):
    dt = datetime.strptime(t, "%H:%M:%S.%f")
    return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6

zone1_time = np.array([time_to_seconds(t) for t in zone1_time])

#zone2_time = np.array([time_to_seconds(t) for t in zone2_time])

#print(len(pow1))
data = np.polyfit(zone2_time, pow2, 2)  #(zone1_time[0:100190], pow1[0:100190], 5
print(data)
# Fit a 5th order polynomial to the data
coefficients = np.polyfit(zone2_time, pow2, 2)  #zone1_time[0:100190], pow1[0:100190], 5

# Generate polynomial function from the coefficients
polynomial = np.poly1d(coefficients)

# Generate x values for plotting the fitted polynomial
x_fit = np.linspace(min(zone2_time), max(zone2_time), 114000)  #104900  #9100
y_fit = polynomial(x_fit)

#Zone1Series1 = [x_fit, y_fit]

# Export x_fit and y_fit to a text file
output_data = np.column_stack((x_fit, y_fit))
#np.savetxt('Zone2Series2_fit.txt', output_data, header='x_fit y_fit', comments='')

# Plot the original data and the fitted polynomial
#plt.scatter(zone1_time, pow1, color='red', label='Data Points')
#plt.plot(x_fit, y_fit, label='5th Order Polynomial Fit')
#plt.xlabel('Time (seconds)')
#plt.ylabel('y')
#plt.legend()
#plt.title('5th Order Polynomial Fit to Data')
#plt.show()

fig0, ax1 = plt.subplots(1,1, constrained_layout = True)
ax1.scatter(zone2_time, pow2, color='red', label='Data Points')
ax1.plot(x_fit, y_fit, label='5th Order Polynomial Fit')
ax1.set(xlabel="Time (s)",ylabel="Power [a.u.]", title = 'Zone2 Series 2')
#ax1.xlabel('Time (seconds)')
#ax1.ylabel('y')
ax1.legend()
#ax1.title('5th Order Polynomial Fit to Data')
#ax1.show()
#print(len(zone1_time))
#def fitting(zone = 'Zone1Series1', Time = zone1_time):
#    X5 = 'x5'
#    X4 = 'x4'
#    X3 = 'x3'
#    X2 = 'x2'
#    X1 = 'x1'
#    C = 'c'
#    
#    ANS1 = Stripe_5.get(zone)
#    ANS2 = Stripe_5.get(zone)
#    ANS3 = Stripe_5.get(zone)
#    ANS4 = Stripe_5.get(zone)
#    ANS5 = Stripe_5.get(zone)
#    ANS6 = Stripe_5.get(zone)
#    if ANS1 is not None:
#        x5 = ANS1.get(X5) #This has been tested and works
#    if ANS2 is not None:
#        x4 = ANS2.get(X4) #This has been tested and works
#    if ANS3 is not None:
#        x3 = ANS1.get(X3) #This has been tested and works
#    if ANS4 is not None:
#        x2 = ANS2.get(X2) #This has been tested and works
#    if ANS5 is not None:
#        x1 = ANS1.get(X1) #This has been tested and works
#    if ANS6 is not None:
#        c = ANS2.get(C) #This has been tested and works
#        
#        return ((x5*time + x4*time + x3*time + x2*time + x1*time + c) for j in Time)

#data1 = fitting(zone = 'Zone1Series1', Time = zone1_time)

x5 = -1.20058556e-15
x4 = 1.19795761e-10
x3 =  4.88184825e-06
x2 = -3.47800999e-01
x1 =  -4.34059516e+04
c  = 2.20974073e+09

data1 = []
for i in zone1_time:
    data1=(x5*zone1_time**(5) + x4*zone1_time**(4) + x3*zone1_time**(3) + x2*zone1_time**(2) + x1*zone1_time + c)
#print(len(data1))
#fig0,ax1 = plt.subplots(1,1,constrained_layout=True)    
#ax1.plot(zone1_time, data1, label=f"Average value for stripe 5 power spectrum")
#ax1.legend()
##ax1.grid()
##ax1.set_ylim([-1800,3500])
##ax1.set_xticks([time[0],time[-1]])
##ax1.set_xticklabels([time[0], time[-1]],rotation = 0, fontsize=9)
#ax1.set(xlabel="Time (s)",ylabel="Strain ($\mu \epsilon$)")
