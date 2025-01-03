# -*- coding: utf-8 -*-
"""
Created on Wed May 29 10:28:16 2024

@author: Hugh Littlehailes
"""

# -*- coding: utf-8 -*-
"""
Aim of file is to determine the temperature compensated strain after the 

@author: Hugh Littlehailes
"""
import numpy as np
import csv
from glob import glob
import matplotlib.pyplot as plt
import math
import time

FibreCoating = {'Stripped': {'S1' : 6.359E-6,   #S1 is the linear temperature sensitivity Units: °C-1 for bare fibre
                             'S2' : 7.947E-9},  #S2 is the linear temperature sensitivity Units: °C-2 for bare fibre
                'ORMOCER':  {'S1' : 8.418E-6,   #Main values - S1 is the linear temperature sensitivity Units: °C-1 for Ormocer coated fibre 
                             'S2' : -3.251E-9}, #Main values - S2 is the linear temperature sensitivity Units: °C-2 for Ormocer coated fibre 
                'ORMOCER-T':{'S1' : 6.910E-6,   #We Don't have this coating - S1 is the linear temperature sensitivity Units: °C-1 for Ormocer-T coated fibre 
                             'S2' : 7.112E-9}   #We Don't have this coating - S2 is the linear temperature sensitivity Units: °C-2 for Ormocer-T coated fibre 
                }
Tref = 22.5     #Fibre calibration temperature
k = 7.56E-7     #Gauge factor
alpha_s = 11.7  # um/m C  - The thermal expansion coefficient of the substrate
alpha_f = 6.5   # um/m C  - The thermal expansion coefficient of the fibre 

def TemperatureProbeCalc(file):
    '''
    Function to extract the time and temperature values from the text file holding the data 
    '''
    T = []
    time = []
    with open(file, 'r') as file:
        
        for _ in range(1):  # Used to exclude data file header strings
            next(file)
        #Separating the data into individual columns
        for line in file:
            columns = line.strip().split('\t')
            T.append(float(columns[3]))
            time.append((columns[2]))
        
    return time, T 


w0 = 1534.962#1534.9623# Stripe 1 
#w0 = 1544.7161# Stripe 2
#w0 = 1524.8984# Stripe 4

T0 = 28.67#26
beta = 0.7704326683329309  #Strain transfer coefficient

def Strain_Formula_Comp(w0_data, w0, TP, T0, S1, S2):
    '''
    Calculates the temperature-compensated strain experienced by the strain FBG 
    This is the amount of mechnical strain experienced by the strain FBG by accounting for the thermal strain affect from ambient temperature, using values from a temperature sensor nearby.
    '''  
    return ((1/k)*(np.log(np.array(w0_data)/w0)-S1*(TP-T0)-S2*((TP-Tref)**2-(T0-Tref)**2))-(alpha_s-alpha_f)*(TP-T0))*(1/beta)
    

def DTG_Strain(file, compensation = True, coating = 'ORMOCER'):
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
    w0_data = []
    time = []
    with open(file, 'r') as file:
        #lines = file.readlines()
        for _ in range(1):
            next(file)
    #Extract the initial wavelength data   
        for line in file:
            #print(line[0:6])
            columns = line.strip().split('\t')
            #print(columns)
            
            time.append((columns[0]))
            w0_data.append((float(columns[1])))
    #        w0_data.append((w0_data1))
        #print(len(columns[4]))
    #w0 = w0_data[0]
    #Extract the corresponding temperature data to compensate with.
    #TP_T1_Out = TemperatureProbeCalc("Substrate2_Grating2_Temp_Reduced_DTG.txt")
    TP_T1_Out = TemperatureProbeCalc("Plate2_Stripe1_Pre-Compensation.txt")
    #TP_T1_Out = TemperatureProbeCalc("Plate2_Stripe2_Pre-Compensation.txt")
    #Define the wavelength data.
    TP_T1_Data = TP_T1_Out[1]
    #Define the 'linenumber' timestamp. 
    times = TP_T1_Out[0]
    #Converts TP_T1_Data to array
    TP = np.array(TP_T1_Data)
    #Define the initial temperature at experiment start.
    T0 = TP[0]
 
        
    result_columns = []
        
    result_columns = Strain_Formula_Comp(w0_data, w0, TP, T0, S1, S2)
    print(result_columns[0])
    fig0,ax1 = plt.subplots(1,1,constrained_layout=True)    
    ax1.plot(time, (result_columns), label=f"T. Compensated Strain Stripe 1 (535 A)")
    ax1.legend()
    ax1.grid()
    #ax1.set_ylim([-1800,3500])
    ax1.set_xticks([time[0],time[-1]])
    ax1.set_xticklabels([time[0], time[-1]],rotation = 0, fontsize=9)
    ax1.set(xlabel="Time (s)",ylabel="Strain ($\mu \epsilon$)")
    
    fig1, ax0 = plt.subplots(1,1,constrained_layout=True)
    ax1 = ax0.twinx()
    ax1.plot(time, TP, 'red')
    ax0.plot(time, w0_data, 'blue' )
    ax0.set_xlabel("Time (s)")
    ax0.set_xticks([times[0],times[-1]])
    ax0.set_ylabel("Wavelength (nm)", color='b')
    ax1.set_ylabel("Temperature (oC)", color='r')
    ax0.set_xticklabels([times[0], times[-1]],rotation = 0, fontsize=9)
    ax0.set_xticks([times[0],times[-1]])
    
    
    
    def export_lists_to_text(list1, list2, filename):
        with open(filename, 'w') as file:
            for item1, item2 in zip(list1, list2):
                file.write(f"{item1},{item2}\n")

    #Output file name
    filename = 'Substrate1_Grating5_CompensatedStrain_Sync.txt'
        
    # Export lists to text file
    #export_lists_to_text(times, result_columns, filename)
    
DTG_Strain("Plate2_Stripe1_Pre-Compensation.txt", coating = 'ORMOCER')
#DTG_Strain("Plate2_Stripe2_Pre-Compensation.txt", coating = 'ORMOCER')


