# -*- coding: utf-8 -*-
"""
Created on Wed May 22 08:54:03 2024

@author: Hugh Littlehailes
"""

import numpy as np
import csv
from glob import glob
import matplotlib.pyplot as plt
import math
import time


def get_Strain(file):
    time_s = []
    strain = []
    with open(file, 'r') as file:
        #Seperate the machine data from the detector data
        for _ in range(1):
            next(file)
        #Separating the data into individual columns
        for line in file:
            columns = line.strip().split('\t')
            time_s.append(columns[0])
            strain.append(float(columns[1]))
    return time_s, strain


def get_Temp(file):
    time_t = []
    temp = []
    with open(file, 'r') as file:
        #Seperate the machine data from the detector data
        for _ in range(1):
            next(file)
        #Separating the data into individual columns
        for line in file:
            columns = line.strip().split('\t')
            time_t.append(columns[0])
            temp.append(float(columns[1]))
    return time_t, temp

time_s, strain = get_Strain("G:/My Drive/LSBU_Trip_Mar2024/S1_G3_TC_StrainData.txt")
time_t, temp = get_Temp("G:/My Drive/LSBU_Trip_Mar2024/S1_G3_TC_TempData.txt")

fig, ax1 = plt.subplots(1,1, constrained_layout = True )
ax2 = ax1.twinx()
ax1.plot(time_s, strain)
ax2.plot(time_t, temp)
