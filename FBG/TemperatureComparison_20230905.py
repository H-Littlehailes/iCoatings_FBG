import numpy as np
import csv
from glob import glob
import matplotlib.pyplot as plt
import math

TC_LINENUMBER = []
TC_WAVELENGTH_FBG1 = []
TC_WAVELENGTH_FBG2 = []
TC_WAVELENGTH_FBG3 = []
TC_PWR1_1 = []
TC_PWR1_2 = []
TC_PWR1_3 = []
TC_Date = 0
TC_Time = 1
TC_LineNumber = 2 
TC_System_status = 3
TC_Wav1_1 = 7
TC_Wav1_2 = 8
TC_Wav1_3 = 9
TC_Pow1_1 = 10
TC_Pow1_2 = 11
TC_Pow1_3 = 12

TC_Tref = 22.5
TC_S1_1 = 7.34295E-06
TC_S1_2 = 1.79377E-09
TC_Lambda_ref1 = 1514.97

TC_S2_1 = 7.40133E-06
TC_S2_2 = 2.21024E-09
TC_Lambda_ref2 = 1524.953

TC_S3_1 = 6.91533E-06
TC_S3_2 = 5.99768E-09
TC_Lambda_ref3 = 1534.937



TC_T1=[]
TC_T2=[]
TC_T3=[]
TC_A=[] 

##TP01
TP_LINENUMBER = []
TP_WAVELENGTH_FBG1 = []
TP_WAVELENGTH_FBG2 = []
TP_WAVELENGTH_FBG3 = []
TP_PWR1_1 = []
TP_PWR1_2 = []
TP_PWR1_3 = []


TP_Date = 0
TP_Time = 1
TP_LineNumber = 2 
TP_System_status = 3
TP_Wav2_1 = 7
TP_Pow2_1 = 8


TP_Tref = 22.5
TP_S1_1 = 6.4153E-06
TP_S1_2 = 7.7948E-09
TP_Lambda_ref1 = 1562.588


TP_T1=[]
TP_A=[]


def TemperatureChainCalc(path):
    with open(path, 'r') as file:
        #Seperate the machine data from the detector data
        for _ in range(18):
            next(file)
        #Separating the data into individual columns
        for line in file:
            columns = line.strip().split()
    
            TC_LINENUMBER.append(float(columns[TC_LineNumber])*0.01)
            #PWR1_1.append(float(columns[Pow1_1]))
            #PWR1_2.append(float(columns[Pow1_2]))
            #pw13 = float(columns[Pow1_3])
            #WAVELENGTH_FBG1.append(float(columns[Wav1_1]))
            #WAVELENGTH_FBG2.append(float(columns[Wav1_2]))
            TC_Wav_1 = float(columns[TC_Wav1_1])
            TC_Wav_2 = float(columns[TC_Wav1_2])
            TC_Wav_3 = float(columns[TC_Wav1_3])
            
            #The equation for converting the shifted wavelength in the output data into to corresponding temperature
            t1 = TC_Tref - (TC_S1_1/(2*TC_S1_2)) + np.sqrt((TC_S1_1/(2*TC_S1_2))**2+(1/TC_S1_2)*np.log(TC_Wav_1/TC_Lambda_ref1))
            t2 = TC_Tref - (TC_S2_1/(2*TC_S2_2)) + np.sqrt((TC_S2_1/(2*TC_S2_2))**2+(1/TC_S2_2)*np.log(TC_Wav_2/TC_Lambda_ref2))
            t3 = TC_Tref - (TC_S3_1/(2*TC_S3_2)) + np.sqrt((TC_S3_1/(2*TC_S3_2))**2+(1/TC_S3_2)*np.log(TC_Wav_3/TC_Lambda_ref3))
            
            TC_WAVELENGTH_FBG1.append(float(TC_Wav_1))
            TC_WAVELENGTH_FBG2.append(float(TC_Wav_2))
            TC_WAVELENGTH_FBG3.append(float(TC_Wav_3))
            
            TC_T1.append(t1)
            TC_T2.append(t2)
            TC_T3.append(t3)
            # print(T)
    return TC_LINENUMBER, TC_T1, TC_T2, TC_T3

def TemperatureProbeCalc(path):
    with open(path, 'r') as file:
         #Seperate the machine data from the detector data
         for _ in range(16):
             next(file)
         #Separating the data into individual columns
         for line in file:
             columns = line.strip().split()
         
             TP_LINENUMBER.append(float(columns[TP_LineNumber])*0.01)
             #PWR1_1.append(float(columns[Pow1_1]))
             #PWR1_2.append(float(columns[Pow1_2]))
             #pw13 = float(columns[Pow1_3])
             #WAVELENGTH_FBG1.append(float(columns[Wav1_1]))
             #WAVELENGTH_FBG2.append(float(columns[Wav1_2]))
             TP_Wav_1 = float(columns[TP_Wav2_1])
             
             
             #The equation for converting the shifted wavelength in the output data into to corresponding temperature
             TP_t1 = TP_Tref - (TP_S1_1/(2*TP_S1_2)) + np.sqrt((TP_S1_1/(2*TP_S1_2))**2+(1/TP_S1_2)*np.log(TP_Wav_1/TP_Lambda_ref1))
             
             
             TP_WAVELENGTH_FBG1.append(float(TP_Wav_1))
                 
             TP_T1.append(TP_t1)
            
        # print(T)
         
        # for i in PWR1_3:
         #fig0,ax1 = plt.subplots(constrained_layout=True)    
         #ax1.plot(TP_LINENUMBER, TP_T1)
         #ax1.set_title('FBG1 1562 nm')
         #ax1.set(xlabel="Time (cs)",ylabel="Temperature ($^oC$)")
         #ax.plot(LINENUMBER, T)
         #fig0.show()
    return TP_LINENUMBER, TP_T1
         

TP_Output = []
TC_Output = []

TC_Output = TemperatureChainCalc("20230920 122244-ILLumiSense-Wav-CH1_Grating_2.txt")
TP_Output = TemperatureProbeCalc("20230920 122244-ILLumiSense-Wav-CH2_Grating_2.txt")

#Temperature_Diff = TC_Output[3] - TP_Output[1]

Temperature_Diff = []    #Calculating the measurement discrepancy between sensors
#TPT1=np.array(TP_Output[1])
#TCT3=np.array(TC_Output[3])
#print(TC_Output[3])
for i in TC_Output:
    TPT1=np.array(TP_Output[1])
    TCT3=np.array(TC_Output[2])
    #TPT1.append(TP_Output[1])
    #TCT3.append(TC_Output[3])
    T_diff = TCT3 - TPT1
print(max(T_diff), min(T_diff))



fig0,(ax1,ax2,ax3) = plt.subplots(1,3,constrained_layout=True)    
ax1.plot(TC_Output[0], TC_Output[1])
#ax1.set_ylim(20,40)
ax1.set_title('FBG1 1515 nm')
ax2.plot(TC_Output[0], TC_Output[2])
#ax2.set_ylim(20,40)
ax2.set_title('FBG2 1525 nm')
ax3.plot(TC_Output[0], TC_Output[3])
#ax3.set_ylim(20,40)
ax3.set_title('FBG3 1535 nm')
ax1.set(xlabel="Time (s)",ylabel="Temperature ($^oC$)")
ax2.set(xlabel="Time (s)",ylabel="Temperature ($^oC$)")
ax3.set(xlabel="Time (s)",ylabel="Temperature ($^oC$)")    
#fig0.show()

fig1,(ax1,ax2) = plt.subplots(2,1,constrained_layout=True)    
ax1.plot(TP_Output[0], TP_Output[1], label="Temperature probe")
ax1.plot(TC_Output[0], TC_Output[2], label="Temperature chain")
ax1.set_title('TP vs TC')
ax1.legend()
ax1.set(xlabel="Time (s)",ylabel="Temperature ($^oC$)")
ax2.plot(TP_Output[0], T_diff)
ax2.set(xlabel="Time (s)",ylabel="Temperature discrepancy ($^oC$)")
fig1.show()
