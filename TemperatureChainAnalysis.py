import numpy as np
import csv
from glob import glob
import matplotlib.pyplot as plt
import math

FILE_DATA = {}
FIBRE_INDEX = {}

DATA = {}

LINENUMBER = []
WAVELENGTH_FBG1 = []
WAVELENGTH_FBG2 = []
WAVELENGTH_FBG3 = []
PWR1_1 = []
PWR1_2 = []
PWR1_3 = []


Date = 0
Time = 1
LineNumber = 2 
System_status = 3
Wav1_1 = 7
Wav1_2 = 8
Wav1_3 = 9
Pow1_1 = 10
Pow1_2 = 11
Pow1_3 = 12

Tref = 22.5
S1_1 = 7.34295E-06
S1_2 = 1.79377E-09
Lambda_ref1 = 1514.97

S2_1 = 7.40133E-06
S2_2 = 2.21024E-09
Lambda_ref2 = 1524.953

S3_1 = 6.91533E-06
S3_2 = 5.99768E-09
Lambda_ref3 = 1534.937



T1=[]
T2=[]
T3=[]
A=[] 

with open("20230830 144346-ILLumiSense-Wav-CH1.txt", "r") as file:
   #Returns the equipment data at the start of each text file, including:
       #Date Time, Equipment type, Equipment ID number, Software version, 
       #  Active channel, Integration Time [µs], High Sensitivity condition,
       #  Sample rate, Data interleave number, Noise Threshold, Any Comment,
    
    line_number = 0 #Start position
    for i in range(11): #End position of equipment data list
            line = file.readline().strip()  #Format data, stripping blank space, and reading it in line-by-line
            FILE_DATA[line_number] = line   #Assigning read line value to a position in the empty data list
            line_number += 1                #Increase counter to iterate to range length
#for key, value in FILE_DATA.items():       #Output result
#    print(f'{value}')
    
    
with open("20230830 144346-ILLumiSense-Wav-CH1.txt", "r") as file:
    #Returns the fibre grating index and corresponding peak wavelength at calibration temperature (~22.5 oC)
    line_number = 13  #Start position of indexing information
    
    for i in range(16):  #End position of indexing information
            line = file.readline().strip()  #Format data, stripping blank space, and reading it in line-by-line
            FIBRE_INDEX[line_number] = line #Assigning read line value to a position in the empty data list
            line_number += 1                #Increase counter to iterate to range length
#for key, value in FIBRE_INDEX.items():     #Output result
 #   print(f'{value}')

#Isolating the data from the file    
#with open("20230830 144346-ILLumiSense-Wav-CH1.txt", "r") as file:
    
  
#    line_number = 18
    
#    for i in range(20):
#            line = file.readline().strip()
#        
#            FIBRE_INDEX[line_number] = line
#            line_number += 1
#for key, value in FIBRE_INDEX.items():
#    print(f'{value}')

#def Temperature_calc(x):
#    return Tref - (S1/(2*S2)) + np.sqrt((S1/(2*S2))**2 +(1/S2)*np.log(x/Lambda_ref))


#Extracting the data to individual matrices    
with open("20230905 115029-ILLumiSense-Wav-CH1.txt", 'r') as file:
    #Seperate the machine data from the detector data
    for _ in range(18):
        next(file)
    #Separating the data into individual columns
    for line in file:
        columns = line.strip().split()
    
        LINENUMBER.append(float(columns[LineNumber]))
        #PWR1_1.append(float(columns[Pow1_1]))
        #PWR1_2.append(float(columns[Pow1_2]))
        #pw13 = float(columns[Pow1_3])
        #WAVELENGTH_FBG1.append(float(columns[Wav1_1]))
        #WAVELENGTH_FBG2.append(float(columns[Wav1_2]))
        Wav_1 = float(columns[Wav1_1])
        Wav_2 = float(columns[Wav1_2])
        Wav_3 = float(columns[Wav1_3])
        
        #The equation for converting the shifted wavelength in the output data into to corresponding temperature
        t1 = Tref - (S1_1/(2*S1_2)) + np.sqrt((S1_1/(2*S1_2))**2+(1/S1_2)*np.log(Wav_1/Lambda_ref1))
        t2 = Tref - (S2_1/(2*S2_2)) + np.sqrt((S2_1/(2*S2_2))**2+(1/S2_2)*np.log(Wav_2/Lambda_ref2))
        t3 = Tref - (S3_1/(2*S3_2)) + np.sqrt((S3_1/(2*S3_2))**2+(1/S3_2)*np.log(Wav_3/Lambda_ref3))
        
        WAVELENGTH_FBG1.append(float(Wav_1))
        WAVELENGTH_FBG2.append(float(Wav_2))
        WAVELENGTH_FBG3.append(float(Wav_3))
    
        T1.append(t1)
        T2.append(t2)
        T3.append(t3)
   # print(T)
    
   # for i in PWR1_3:
    fig0,(ax1,ax2,ax3) = plt.subplots(1,3,constrained_layout=True)    
    ax1.plot(LINENUMBER, T1)
    ax1.set_title('FBG1 1515 nm')
    ax2.plot(LINENUMBER, T2)
    ax2.set_title('FBG2 1525 nm')
    ax3.plot(LINENUMBER, T3)
    ax3.set_title('FBG3 1535 nm')
    ax1.set(xlabel="Time (cs)",ylabel="Temperature ($^oC$)")
    ax2.set(xlabel="Time (cs)",ylabel="Temperature ($^oC$)")
    ax3.set(xlabel="Time (cs)",ylabel="Temperature ($^oC$)")
    #ax.plot(LINENUMBER, T)
    fig0.show()    
    
    
    

    
    
    
