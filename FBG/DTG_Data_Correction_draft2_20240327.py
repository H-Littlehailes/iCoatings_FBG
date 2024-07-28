# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:35:43 2024

@author: Hugh Littlehailes
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:11:24 2024

@author: Hugh Littlehailes
"""

import pandas as pd
import numpy as np


with open("G:/My Drive/LSBU_Trip_Mar2024/Second_Substrate/20240321 184728-ILLumiSense-Wav-CH2_Reduced.txt", 'r') as file:
    lines = file.readlines()

# Define a threshold for the difference in lengths of rows
difference_threshold = 1

# Iterate over each row to detect and correct skipped values
corrected_lines = []
corrected_line = []
for i in range(1, len(lines)):
    prev_row = lines[i-1].strip().split('\t')
    curr_row = lines[i].strip().split('\t')
    #if len(prev_row) - len(curr_row) >0:
        # Missing values detected, shift values accordingly
    
    for j in range(len(curr_row)):
            if j == 0:
                corrected_line[0] = curr_row[0]        
            #if (abs(curr_row[j]-prev_row[j])/prev_row[j])>0.01:
            #    corrected_line +=
            if float(curr_row[j]) > 1554: 
                corrected_line[7] = curr_row[j]
            if float(1546<curr_row[j])<1554:
                corrected_line[6] = curr_row[j]
            if float(1536<curr_row[j])<1544:
                corrected_line[5] = curr_row[j]
            if float(1526<curr_row[j])<1534:
                corrected_line[4] = curr_row[j]
            if float(1516<curr_row[j])<1524:
                corrected_line[3] = curr_row[j]
            if float(1000<curr_row[j])<1514:
                corrected_line[2] = curr_row[j]
            else:
                corrected_line[j] += 0
           
    #    corrected_lines.append(corrected_line + '\n')
    #else:
    #    corrected_lines.append(lines[i])

   # Write the corrected data back to a new text file
output_file_path = 'corrected_data_test.txt'
with open(output_file_path, 'w') as output_file:
    output_file.writelines(corrected_lines)     
                
                
                
                
                
                
                
                
                
                
                
                
                
                
