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
Wav2_1 = 7
Pow2_1 = 8


Tref = 22.5
S1_1 = 6.4153E-06
S1_2 = 7.7948E-09
Lambda_ref1 = 1562.588


T1=[]
A=[] 


with open("20230905 115029-ILLumiSense-Wav-CH2.txt", "r") as file:
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
    
    
with open("20230905 115029-ILLumiSense-Wav-CH2.txt", "r") as file:
    #Returns the fibre grating index and corresponding peak wavelength at calibration temperature (~22.5 oC)
    line_number = 12  #Start position of indexing information
    
    for i in range(14):  #End position of indexing information
            line = file.readline().strip()  #Format data, stripping blank space, and reading it in line-by-line
            FIBRE_INDEX[line_number] = line #Assigning read line value to a position in the empty data list
            line_number += 1                #Increase counter to iterate to range length
for key, value in FIBRE_INDEX.items():     #Output result
    print(f'{value}')
 
 #Extracting the data to individual matrices    
with open("20230905 115029-ILLumiSense-Wav-CH2.txt", 'r') as file:
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
         Wav_1 = float(columns[Wav2_1])
         
         
         #The equation for converting the shifted wavelength in the output data into to corresponding temperature
         t1 = Tref - (S1_1/(2*S1_2)) + np.sqrt((S1_1/(2*S1_2))**2+(1/S1_2)*np.log(Wav_1/Lambda_ref1))
         
         
         WAVELENGTH_FBG1.append(float(Wav_1))
             
         T1.append(t1)
        
    # print(T)
     
    # for i in PWR1_3:
     fig0,ax1 = plt.subplots(constrained_layout=True)    
     ax1.plot(LINENUMBER, T1)
     ax1.set_title('FBG1 1562 nm')
     ax1.set(xlabel="Time (cs)",ylabel="Temperature ($^oC$)")
     #ax.plot(LINENUMBER, T)
     fig0.show()    
