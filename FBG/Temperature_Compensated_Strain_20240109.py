# -*- coding: utf-8 -*-
"""
Created on Mon Jan  8 16:45:57 2024

@author: Hugh Littlehailes
"""
import numpy as np
import csv
from glob import glob
import matplotlib.pyplot as plt
import math

#The key fibre parameters
Tref = 22.5
k = 7.59E-7
S1 = 6.353E-06
S2 = 7.894E-09
alpha_s = 0
alpha_f = 0.5

def TemperatureProbeCalc(file):
    TP_LINENUMBER = [] 
    TP_LineNumber = 2 
    TP_Wav2_1 = 7
    TP_Tref = 22.5
    TP_S1_1 = 6.4153E-06
    TP_S1_2 = 7.7948E-09
    TP_Lambda_ref1 = 1562.588
    TP_WAVELENGTH_FBG1 = []
    TP_T1=[]
    with open(file, 'r') as file:
         #Seperate the machine data from the detector data
         for _ in range(16):
             next(file)
         #Separating the data into individual columns
         for line in file:
             columns = line.strip().split()
             TP_LINENUMBER.append(float(columns[TP_LineNumber])*0.01)
             
             TP_Wav_1 = float(columns[TP_Wav2_1])
             #The equation for converting the shifted wavelength in the output data into to corresponding temperature
             TP_t1 = TP_Tref - (TP_S1_1/(2*TP_S1_2)) + np.sqrt((TP_S1_1/(2*TP_S1_2))**2+(1/TP_S1_2)*np.log(TP_Wav_1/TP_Lambda_ref1))
             
             TP_WAVELENGTH_FBG1.append(float(TP_Wav_1))                 
             TP_T1.append(TP_t1)                    
    return TP_LINENUMBER, TP_T1


TP_T1_Out = TemperatureProbeCalc("20230927 152254-ILLumiSense-Wav-CH4.txt")
TP_T1_Data = TP_T1_Out[1]
LINENUMBER = TP_T1_Out[0]

def Strain_Formula_Comp(columns, w0_data, TP, T0, S1, S2):
    '''
    Calculates the temperature-compensated strain experienced by the strain FBG 
    This is the amount of mechnical strain experienced by the strain FBG by accounting for the thermal strain affect from ambient temperature, using values from a temperature sensor nearby.
    '''
    return [(1/k)*(math.log(Wavelength/w0_data))-S1*(TP-T0)-S2*((TP-22.5)**2-(T0-Tref)**2)-(alpha_s-alpha_f)*(TP-T0) for i, Wavelength in enumerate(columns)]

def Strain_Formula(columns, w0_data):
    '''
    Calculates the uncompensated strain measured by the strain FBG.
    This does not account for the strain induced by ambient temperature.
    epsilon = mechanical strain + thermal strain.
    '''
    return [(1/k)*(math.log(wavelength/w0_data)) for i, wavelength in enumerate(columns)]

def Temperature_Compensated_Strain(file, F_s = 100, compensation = True, file_size_large = True, n = 1000):
    '''
    Function to determine the strain measured within the gratings of the DTG fibres.
        - F_s is the sampling frequency of the recording in Hz, also found in line 8 of the interrogator txt file.
        - Compensation condition when true allows for the calculation of the temperature compensated strain.
        - If the file size is large and calculation heavy, the file_size_large condition allows for a reduced 
          set to be formed by selecting every nth data value.
    
    '''
    with open(file, "r") as file:
    	lines = file.readlines()
    #Extract the initial wavelength data   
    w0_data = [int(float(line.strip().split('\t')[1])) for line in lines[13:14]]  
	#Extract the corresponding temperature data to compensate with.
    TP_T1_Out = TemperatureProbeCalc("20230927 152254-ILLumiSense-Wav-CH4.txt")   
    #Define the wavelength data.
    TP = TP_T1_Out[1]      
    #Define the 'linenumber' timestamp.                                                       
    LINENUMBER = TP_T1_Out[0]  
    #Define the initial temperature at experiment start.                                                   
    T0 = TP[0]                                                                    
    
    #Extract the wavelength data from the strain fibre file.
    selected_data = [line.strip().split('\t')[4:5] for line in lines[16:]]	  
    #Float the data
    selected_data = [[float(value) for value in row] for row in selected_data]
    #Extract the linenumber timestamp and accounting for sampling frequency convert to seconds for plotting
    x_data = [int(float(line.strip().split('\t')[2]))*(1/F_s)  for line in lines[16:]]
    
    #Account for temperature-compensation
    if compensation == True:
        #If it is a large file, reduce the data resolution for every nth value for quicker calculation and plotting
        if file_size_large == True:        
            Wav_Strain_Reduced = selected_data[::n]
            TP_Reduced = TP_T1_Data[::n]
            x_data_Reduced = x_data[::n]
            #Apply the formula to the reduced data
            result_columns = [Strain_Formula_Comp(column,w0_data,TP_Reduced, T0, S1, S2) for column, w0_data, TP_Reduced in zip(zip(*Wav_Strain_Reduced), w0_data, TP_Reduced)]
        else:
        	# Apply the formula to each column
        	result_columns = [Strain_Formula_Comp(column,w0_data,TP, T0, S1, S2) for column, w0_data, TP in zip(zip(*selected_data), w0_data, TP)]
    else:
        result_columns = [Strain_Formula(column,w0_data) for column, w0_data, in zip(zip(*selected_data), w0_data)]
        
    # Print the result
    for i, result_column in enumerate(result_columns):
        if file_size_large == True:
            fig0,(ax1) = plt.subplots(1,1,constrained_layout=True)    
            ax1.plot(x_data_Reduced, result_column, label=f"Temperature Compensated Strain Grating {w0_data} nm")
            ax1.legend()
            ax1.set(xlabel="Time (s)",ylabel="Strain ($\mu \epsilon$)")
        else:    
            fig0,(ax1) = plt.subplots(1,1,constrained_layout=True)    
            ax1.plot(x_data, result_column, label=f"Temperature Compensated Strain Grating {w0_data} nm")
            ax1.legend()
            ax1.set(xlabel="Time (s)",ylabel="Strain ($\mu \epsilon$)")

    
Temperature_Compensated_Strain("20230927 152254-ILLumiSense-Wav-CH2.txt", F_s = 100, compensation = True, file_size_large = True, n = 100)
