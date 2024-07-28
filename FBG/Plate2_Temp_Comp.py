# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 14:00:56 2024

@author: Hugh Littlehailes
"""

import numpy as np
import csv
from glob import glob
import matplotlib.pyplot as plt
import math
import time
from datetime import datetime

FibreCoating = {'Stripped': {'S1' : 6.359E-6,
                             'S2' : 7.947E-9},
                'ORMOCER':  {'S1' : 8.418E-6,
                             'S2' : -3.251E-9},
                'ORMOCER-T':{'S1' : 6.910E-6,
                             'S2' : 7.112E-9}
                }
Tref = 22.5
k = 7.56E-7
alpha_s = 11.7#e-6
alpha_f = 6.5#e-6
S_1 = []

w0 = [1514.9821,
      1524.8984,
      1534.9623,
      1544.7161,
      1555.0741]


file = "G:/My Drive/LSBU_Trip_Mar2024/Plate2_Data_2.txt"

def time_to_seconds(t):
    dt = datetime.strptime(t, "%H:%M:%S")
    return dt.hour * 3600 + dt.minute * 60 + dt.second

def time_to_seconds_long(t):
    dt = datetime.strptime(t, "%H:%M:%S.%f")
    return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6

def TemperatureProbeCalc(file, t_index, index, start, end):     
    """
    Reads data from a text file and adds data between specific rows to a list.
    
    Parameters:
    file (str): Path to the text file.
    start (int): Line number to start reading (inclusive).
    end (int): Line number to stop reading (inclusive).
    
    Returns:
    list: List of lines between the specified rows.
    """
    T = []
    Time = []
    with open(file, 'r') as file:
        #Seperate the machine data from the detector data
        for _ in range(2):
            next(file)
        for current_line_number, line in enumerate(file, start=1):
            if start <= current_line_number <= end:
                columns = line.strip().split('\t')
                T.append(float(columns[index]))
                Time.append((columns[t_index]))
                
    return Time, T 



time_T1, T1 = TemperatureProbeCalc(file, t_index = 2, index = 3, start = 0, end = 961)
time_T2, T2 = TemperatureProbeCalc(file, t_index = 6, index = 7, start = 0, end = 690)
time_T4, T4 = TemperatureProbeCalc(file, t_index = 10, index = 11, start = 0, end = 763)
#time_T5, T5 = TemperatureProbeCalc(file, t_index = 8, index = 11, start = 3365, end = 4295)

#1 = float(T1)
T1 = [float(str(s)) for s in T1]
#T1 = float(str(T1))
print(type(T1))
print(T1[0])
time_T1 = np.array([time_to_seconds(t) for t in time_T1])
time_T2 = np.array([time_to_seconds(t) for t in time_T2])
time_T4 = np.array([time_to_seconds(t) for t in time_T4])
#time_T5 = np.array([time_to_seconds(t) for t in time_T5])

time_St1, St1 = TemperatureProbeCalc(file, t_index = 0, index = 1, start = 0, end = 961)
time_St2, St2 = TemperatureProbeCalc(file, t_index = 4, index = 5, start = 0, end = 690)
time_St4, St4 = TemperatureProbeCalc(file, t_index = 8, index = 9, start = 0, end = 763)
#time_St5, St5 = TemperatureProbeCalc(file, t_index = 0, index = 4, start = 3365, end = 4295)

time_St1 = np.array([(time_to_seconds_long(t)) for t in time_St1])
time_St2 = np.array([(time_to_seconds_long(t)) for t in time_St2])
time_St4 = np.array([(time_to_seconds_long(t)) for t in time_St4])
#time_St5 = np.array([(time_to_seconds_long(t)-10) for t in time_St5])

fig1, ax0 = plt.subplots(1,1,constrained_layout=True)
ax1 = ax0.twinx()
ax0.plot(time_T1, T1, 'red')
ax0.plot(time_T2, T2, 'blue')
ax0.plot(time_T4, T4, 'green')
#ax0.plot(time_T5, T5, 'black')
ax0.set_xlabel("Time (s)")
ax1.plot(time_St1, St1, 'red')
ax1.plot(time_St2, St2, 'blue')
ax1.plot(time_St4, St4, 'green')
#ax1.plot(time_St5, St5, 'black')
#ax0.set_xlabel("Time (s)")
ax1.set_ylabel("Wavelength (nm)", color='b')
ax0.set_ylabel("Temperature (oC)", color='r')
#ax0.set_xticklabels([times[0], times[-1]],rotation = 0, fontsize=9)
#ax0.set_xticks([times[0],times[-1]])

time_1 = np.linspace(0, 961, 961)
time_2 = np.linspace(0, 690, 690)
time_4 = np.linspace(0, 763, 763)
#time_5 = np.linspace(0, 931, 931)

T0 = 25.7817


def Strain_Formula_Comp(w0_data, w0, TP, T0, S1, S2):
    '''
    Calculates the temperature-compensated strain experienced by the strain FBG 
    This is the amount of mechnical strain experienced by the strain FBG by accounting for the thermal strain affect from ambient temperature, using values from a temperature sensor nearby.
    ''' 
    TP = np.array(TP)
    return ((1/k)*(np.log(np.array(w0_data)/w0)-S1*(TP-T0)-S2*((TP-Tref)**2-(T0-Tref)**2))-(alpha_s-alpha_f)*(TP-T0))*(1/0.55577141)
    

def DTG_Strain(Stripe_Strain, Stripe_Temp, Time, w0_index, compensation = True, coating = 'ORMOCER'):
    '''
    Function to determine the strain measured within the gratings of the DTG fibres.
    F_s is the sampling frequency of the recording in Hz, also found in line 8 of the interrogator txt file.
    Compensation condition when true allows for the calculation of the temperature compensated strain.
    The coating refers to the material surrounding the strain fibre which will affect the linear and quadratic factors.
        The coating options are: stripped, ORMOCER, and ORMOCER-T, as defined in FibreCoating dictionary at the top of the page.
        Values are provided from the data sheets from FBGS
    '''
    strain_value1 = 'S1'
    strain_value2 = 'S2'
    ANS1 = FibreCoating.get(coating)
    ANS2 = FibreCoating.get(coating)
    if ANS1 is not None:
        S1 = ANS1.get(strain_value1) #This has been tested and works
    if ANS2 is not None:
        S2 = ANS2.get(strain_value2) #This has been tested and works

        
    result_columns = []
        
    result_columns = Strain_Formula_Comp(Stripe_Strain, w0[w0_index], Stripe_Temp, T0, S1, S2)# for i, j, in zip(Stripe_Strain, Stripe_Temp)]
    print(result_columns[0])
    fig0,ax1 = plt.subplots(1,1,constrained_layout=True)    
    ax1.plot(time_4, (result_columns)-result_columns[0], label=f"T. Compensated Strain Stripe 4 (550 A)")
    ax1.legend()
    ax1.grid()
    #ax1.set_ylim([-1800,3500])
    #ax1.set_xticks([time[0],time[-1]])
    #ax1.set_xticklabels([time[0], time[-1]],rotation = 0, fontsize=9)
    ax1.set(xlabel="Time (s)",ylabel="Strain ($\mu \epsilon$)")
    
    fig1, ax0 = plt.subplots(1,1,constrained_layout=True)
    ax1 = ax0.twinx()
    ax1.plot(time_T4, Stripe_Temp, 'red')
    ax0.plot(time_St4, Stripe_Strain, 'blue' )
    ax0.set_xlabel("Time (s)")
    #ax0.set_xticks([Time[0],Time[-1]])
    ax0.set_ylabel("Wavelength (nm)", color='b')
    ax1.set_ylabel("Temperature (oC)", color='r')
    #ax0.set_xticklabels([Time[0], Time[-1]],rotation = 0, fontsize=9)
    #ax0.set_xticks([times[0],times[-1]])
    result_columns = result_columns-result_columns[0]
    
    def export_lists_to_text(list1, list2, filename):
        with open(filename, 'w') as file:
            for item1, item2 in zip(list1, list2):
                file.write(f"{item1},{item2}\n")

    #Output file name
    filename = 'Substrate2_Stripe4_CompensatedStrain_Origin.txt'
        
    # Export lists to text file
    #export_lists_to_text(time_2, result_columns, filename)
    
#DTG_Strain("Substrate2_Grating1_Strain_Sync.txt", coating = 'ORMOCER')
#DTG_Strain("Plate2_Stripe2_Pre-Compensation.txt", coating = 'ORMOCER')
DTG_Strain( St4, T4, time_T4, w0_index = 1,  compensation = True, coating = 'ORMOCER')
