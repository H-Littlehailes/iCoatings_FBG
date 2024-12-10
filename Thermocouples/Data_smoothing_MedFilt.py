# -*- coding: utf-8 -*-
"""
Created on Fri May 24 12:28:41 2024

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

Time = []
H1 = []
H2 = []
H3 = []
H4 = []
H5 = []
Stripe_5 = []
Pow_2 = []

file_1 = "G:/My Drive/LSBU_Trip_Mar2024/Second_Substrate/03211835.txt"
file_2 = "G:/My Drive/LSBU_Trip_Mar2024/P2_S5_G1_Pow2.txt"
file_P2_S5_Wav = "G:/My Drive/LSBU_Trip_Mar2024/P2_S5_G1_Average_Wav.txt"

def extract_time (file):
    
    with open(file_P2_S5_Wav) as file:
        for _ in range(1):
            next(file)
        for line in file:
            columns = line.strip().split('\t')
            Time.append((columns[0]))
    return Time

Time = extract_time(file_P2_S5_Wav)

def extract (file, index):
    val = []
    with open(file) as file:
        for _ in range(1):
            next(file)
        for line in file:
            columns = line.strip().split('\t')
            val.append(float(columns[index]))
       
    return val
Stripe_5 = extract(file_P2_S5_Wav, index = 1)
Pow_2 = extract(file_2, index = 1)
#H1 = extract(file_1, index = 2)
#H2 = extract(file_1, index = 4)
#H3 = extract(file_1, index = 6)
#H4 = extract(file_1, index = 8)
#H5 = extract(file_1, index = 10)

Pow_2_Smooth = scipy.signal.medfilt(Pow_2, kernel_size=7001)
Stripe_5_smooth = scipy.signal.medfilt(Stripe_5, kernel_size=71)
#H1_smooth = scipy.signal.medfilt(H1, kernel_size=3) + 81.48970703 #87.00067613 Value average from stable temp values between 18:42:00-18:44:00
#H2_smooth = scipy.signal.medfilt(H2, kernel_size=3) + 47.42994023 #45.94091146
#H3_smooth = scipy.signal.medfilt(H3, kernel_size=3) + 88.53930938
#H4_smooth = scipy.signal.medfilt(H4, kernel_size=3) + 81.77456016 #83.13285
#H5_smooth = scipy.signal.medfilt(H5, kernel_size=3) + 39.00435469 #41.53197958
    
time_axis_min = Time[0]
time_axis_max = Time[-1]
#fig, axs = plt.subplots(2,3, constrained_layout = True)
fig, axs = plt.subplots( constrained_layout = True)
    
    
axs.plot(Time, Pow_2)
axs.set_title('TC_5_Moving_Average')
axs.set(xlabel = "Time", ylabel = "Temperature (oC)")
#axs.set_ylim([0,200])
#axs.set_xticks([Time[0],Time[-1]])
#axs.set_xticklabels([time_axis_min, time_axis_max],rotation = 0, fontsize=9)
plt.show()
    
#with open("Substrate2_Grating3_Temp.txt", "w") as text_file:
#    for Time, H1_smooth in zip(Time, H1_smooth): 
#        text_file.write(Time, H1_smooth)    
    
def export_lists_to_text(list1, list2, filename):
    with open(filename, 'w') as file:
        for item1, item2 in zip(list1, list2):
            file.write(f"{item1},{item2}\n")

    #Output file name
    filename = 'Substrate2_S5_G1_T5_Temp.txt'
        
    # Export lists to text file
    #export_lists_to_text(Time, H5_smooth, filename)
    
