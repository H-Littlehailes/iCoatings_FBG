# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 11:43:31 2024

@author: Hugh Littlehailes

An attempt to correct the clust-f*** of the DTG data
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import csv
from glob import glob
import math

Time = []
Date = []
Warn = []
Wav1 = []
Wav2 = []
Wav3 = []
Wav4 = []
Wav5 = []
Wav6 = []
Wav7 = []
Pow1 = []
Pow2 = []
Pow3 = []
Pow4 = []
Pow5 = []
Pow6 = []
Pow7 = []


with open("G:/My Drive/LSBU_Trip_Mar2024/Second_Substrate/20240321 184728-ILLumiSense-Wav-CH2_20240416.txt",'r') as file:
    for _ in range(22):
         next(file)
    for line in file:
        columns = line.strip().split()
        Time.append((columns[1]))
        Warn.append((columns[3]))
        if Warn == -1:
            Wav1.append('NaN')
            Wav2.append(float(columns[7]))
            Wav3.append(float(columns[8]))
            Wav4.append(float(columns[9]))
            Wav5.append(float(columns[10]))     
            Wav6.append(float(columns[11])) 
            Wav7.append(float(columns[12])) 
            
            Pow1.append('NaN')
            Pow2.append(float(columns[13]))
            Pow3.append(float(columns[14]))
            Pow4.append(float(columns[15]))
            Pow5.append(float(columns[16]))
            Pow6.append(float(columns[17]))
            Pow7.append(float(columns[18]))
        elif Warn == -2:
            
            Wav1.append('NaN')
            Wav2.append(float(columns[7]))
            Wav3.append(float(columns[8]))
            Wav4.append(float(columns[9]))
            Wav5.append(float(columns[10]))     
            Wav6.append(float(columns[11])) 
            Wav7.append('NaN') 
        
            Pow1.append('NaN')
            Pow2.append(float(columns[12]))
            Pow3.append(float(columns[13]))
            Pow4.append(float(columns[14]))
            Pow5.append(float(columns[15]))
            Pow6.append(float(columns[16]))
            Pow7.append('NaN')
        elif Warn == 0:
            Wav1.append(float(columns[7]))
            Wav2.append(float(columns[8]))
            Wav3.append(float(columns[9]))
            Wav4.append(float(columns[10]))
            Wav5.append(float(columns[11]))     
            Wav6.append(float(columns[12])) 
            Wav7.append(float(columns[13])) 
            
            Pow1.append(float(columns[14]))
            Pow2.append(float(columns[15]))
            Pow3.append(float(columns[16]))
            Pow4.append(float(columns[17]))
            Pow5.append(float(columns[18]))
            Pow6.append(float(columns[19]))
            Pow7.append(float(columns[20]))
        
    print((Wav1[0]))
#    for i in range(Wav1[9340], Wav1[9393]):
#            if abs(float(Wav1[i]-float(Wav1[i-1])))>0.04:
#                Pow7[i] = Pow5[i]
#                Pow6[i] = Pow4[i]
#                Pow5[i] = Pow3[i]
#                Pow4[i] = Pow2[i]
#                Pow3[i] = Pow1[i]
#                Pow2[i] = Wav7[i]
#                Pow1[i] = 0#np.nan
#                
#                Wav7[i] = Wav6[i]
#                Wav6[i] = Wav5[i]
#                Wav5[i] = Wav4[i]
#                Wav4[i] = Wav3[i]
#                Wav3[i] = Wav2[i]
#                Wav2[i] = Wav1[i]
#                
                
                
    fig, ax = plt.subplots(constrained_layout = True)
    ax.plot(Time[1:-1:1000], Wav1[1:-1:1000])
