import numpy as np
import csv
from glob import glob
import matplotlib.pyplot as plt
import math
import time


LINENUMBER = []
LINENUMBER1 = []  
wav_0 = []
Wav_Strain = []
STRN1_1 = []
STRN1_2 = []
STRN1_Diff = []
Wav_Strain_Reduced = []
TP_T1_Reduced = []
LINENUMBER_Reduced = []

Date = 0
Time = 1
LineNumber = 2 
System_status = 3
strain = 7


Tref = 22.5
k = 7.59E-7
S1 = 6.353E-06
S2 = 7.894E-09
Lambda_ref1 = 1526.7931

#k = 7.59E-07
lambda_0=[]
alpha_s = 0
alpha_f = 0.5

# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


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

TP_T1_Out = TemperatureProbeCalc("20230927 142306-ILLumiSense-Wav-CH4.txt")
TP_T1_Data = TP_T1_Out[1]
LINENUMBER = TP_T1_Out[0]


def Strain_Values(file):
    with open(file, 'r') as file:
        for line_number, line in enumerate(file):  #Isolating the starting wavelength measured by the software. It is recorded on the same line as the grating number so needs to be separated.
            if line_number == 13:   #Grating ID and start wavelength on this line
                columns = line.strip().split()#Format data, stripping blank space, and reading it in line-by-line
                wav_00 = float(columns[1])            
                wav_0.append(wav_00)
                break    
                      
         
            #Separate the machine data from the detector data. Data starts on line 16 of text file
        for _ in range(16):     
            next(file)
        #Separating the data into individual columns
        for line in file:
            columns = line.strip().split()
            LINENUMBER1.append(float(columns[LineNumber])*0.01)  #Sampling rate default to 100 Hz so x-axis in units centi-seconds, divide by 100 to get in seconds
       # Wav_Strain = (float(columns[strain])) #The wavelengths from the strain fibre
            Wav_Strain.append(float(columns[strain]))
        
        #wav_0 = lambda_0[0]   #Renaming the initial wavelength at experiment start
        #TP_T1 = TP_T1_Data
            TP_T1 = np.array(TP_T1_Data)
            TP_0 = TP_T1[0]
    return LINENUMBER1, Wav_Strain, TP_T1, TP_0, wav_0    

Strain_data = Strain_Values("20230927 142306-ILLumiSense-Wav-CH2.txt")
LINENUMBER1 = Strain_data[0]
Wav_Strain = Strain_data[1]
TP_T1 = Strain_data[2]
TP_0 = Strain_data[3]
wav_0 = Strain_data[4]


## To reduce the processing load reduced variable are generated, selecting every nth value
n = 100  #Defines the data step-count to select every nth value
for i in range(0, len(Wav_Strain),n):
    Wav_Strain1 = Wav_Strain[i]
    Wav_Strain_Reduced.append(float(Wav_Strain1))
    TP_T11 = TP_T1[i]
    TP_T1_Reduced.append(float(TP_T11))
    LINENUMBER11 = LINENUMBER1[i]
    LINENUMBER_Reduced.append(float(LINENUMBER11))
    
#print(type(Wav_Strain_Reduced))
#print(type(TP_T1_Reduced))
#TP_T11 = np.array(TP_T1)
#TP_T1_Reduced = TP_T11[0::n]
l = len(LINENUMBER_Reduced[0::n])
#printProgressBar(0, l, prefix = 'Progress:', suffix = 'Complete', length = 50)

for i in enumerate(Wav_Strain_Reduced):
    Wav_Strain_Reduced = np.array(Wav_Strain_Reduced)
    TP_T1_Reduced = np.array(TP_T1_Reduced)
    
    a = (1/k)
    b = Wav_Strain_Reduced/wav_0
    c = S1*(TP_T1_Reduced-TP_0)
    d = S2*((TP_T1_Reduced-22.5)**2-(TP_0-Tref)**2)
    e = (alpha_s-alpha_f)*(TP_T1_Reduced-TP_0)
    
    epsilon = a*(np.log(b) - c - d) - e
    #epsilon = (1/k)*(np.log(Wav_Strain_Reduced/wav_0)-S1*(TP_T1_Reduced-TP_0)-S2*((TP_T1_Reduced-22.5)**2-(TP_0-Tref)**2))-(alpha_s-alpha_f)*(TP_T1_Reduced-TP_0)
    epsilon_NoComp = (1/k)*np.log(Wav_Strain_Reduced/wav_0)
    
    epsilon_diff = epsilon_NoComp - epsilon
    #time.sleep(0.1)
    #printProgressBar(i + 1, l, prefix = 'Progress:', suffix = 'Complete', length = 50)
    
    
    
STRN1_1.append(epsilon)
STRN1_2.append(epsilon_NoComp)
STRN1_Diff.append(epsilon_diff)

STRN1_1 = np.array(STRN1_1).T
STRN1_2 = np.array(STRN1_2).T
STRN1_Diff = np.array(STRN1_Diff).T
#print(len(TP_T1))    
fig0,(ax1, ax2, ax3) = plt.subplots(3,1,constrained_layout=True)    
ax1.plot(LINENUMBER_Reduced, STRN1_1, label="Temperature Compensated Strain")
ax1.plot(LINENUMBER_Reduced, STRN1_2, label="Uncompensated Strain")
ax1.legend()
ax1.set(xlabel="Time (s)",ylabel="Strain ($\mu \epsilon$)")
ax2.plot(LINENUMBER_Reduced, TP_T1_Reduced, label="Temperature")
ax2.legend()
ax2.set(xlabel="Time (s)",ylabel="Temperature ($^oC$)")
ax3.plot(LINENUMBER_Reduced, STRN1_Diff, label="Strain calculation discrepancy")
ax3.legend()
ax3.set(xlabel="Time (s)",ylabel="Strain Discrepancy")
fig0.savefig("Temperature_Compensated_Strain_20231005.png")
fig0.show()    
