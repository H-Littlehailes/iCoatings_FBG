# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 15:47:25 2024

@author: Hugh Littlehailes
"""

import numpy as np
import csv
from glob import glob
import matplotlib.pyplot as plt
import math
import time

Time = []
Date = []
Warn = []
Col1 = []   # 1515a
Col2 = []   # 1515b
Col3 = []   # 1515c
Col4 = []   # 1515d
Col5 = []   # 1525
Col6 = []   # 1535
Col7 = []   # 1545a
Col8 = []   # 1545b
Col9 = []   # 1555a
Col10 = []   # 1555b
#same_first_three_digits= {}  # Dictionary to store values with the same first three digits

        # Load the data from the text file
with open("G:/My Drive/LSBU_Trip_Mar2024/Second_Substrate/20240321 184728-ILLumiSense-Wav-CH2_20240416.txt",'r') as file:
    #lines = file.readlines()
    for _ in range(22):
         next(file)
    
#    for line in file:
#        columns = line.strip().split()
#        Time.append((columns[1]))
        #Warn.append((columns[3]))
        
    #    if columns[7]<1525:
    #        Col1.append(columns[7])
    #for line in file:
    #    time_value = line.strip().split()
    #    Time.append(time_value[1])
    
    
    lines = file.readlines()    
    for j in lines:
            # Split the line into individual values
        
                   
        values = j.strip().split('\t')
        time_stamp = values[1]
        Time.append(time_stamp)
        # Dictionary to store values with the same first three digits
        #same_digits_columns[lines] = {}
        first_three_digits = {}
        for i in values[4:]: 
            #Time.append((values[1]))
            # Convert value to float
            try:
                float_value = float(i)
                first_three = int(str(int(float_value))[:3])
                # Assign value to appropriate magnitude column
                if abs(float_value) < 1520:
                        if first_three in first_three_digits:
                            # Handle values with the same first three digits
                            if len(first_three_digits[first_three]) == 1:
                                # Two values with the same first three digits
                                Col1.append(min(first_three_digits[first_three][0], float_value))
                                Col3.append(max(first_three_digits[first_three][0], float_value))
                            elif len(first_three_digits[first_three]) == 2:
                                # Three values with the same first three digits
                                sorted_values = sorted(first_three_digits[first_three] + [float_value])
                                Col1.append(sorted_values[0])
                                Col2.append(sorted_values[1])
                                Col3.append(sorted_values[2])
                            del first_three_digits[first_three]  # Clear the dictionary entry
                        else:
                            first_three_digits[first_three] = [float_value]
                    
                    #Col1.append(float_value)
                
                elif 1520 <= abs(float_value) < 1530:
                    Col5.append(float_value)
                elif 1530 <= abs(float_value) < 1540:
                    Col6.append(float_value)
                elif 1540 <= abs(float_value) < 1550:
                    Col7.append(float_value)    
                else:
                    Col9.append(float_value)
            except ValueError:
                   pass  # Ignore non-numeric values
#Time = np.arange(0,578938,1)                   
print(len(Time))                   
#print((Time[0]))    
#fig, ax = plt.subplots(constrained_layout = True)
#ax.plot(Time[1:-1:1000], Col6[1:-1:1000])   
#ax.set(xlabel = "Time", ylabel = "Temperature (oC)")
#ax.set_ylim([1534,1539])
#ax.set_xticks([Time[0],Time[-1]])
#ax.set_xticklabels([Time[0], Time[-1]],rotation = 0, fontsize=9)
#ax.xlabel('Time')
    
def export_lists_to_text(list1, list2, filename):
    with open(filename, 'w') as file:
        for item1, item2 in zip(list1, list2):
            file.write(f"{item1},{item2}\n")



#Output file name
filename = 'Substrate2_Grating2_Strain.txt'
    
# Export lists to text file
#export_lists_to_text(Time, Col5, filename)    
    
    
    
    
