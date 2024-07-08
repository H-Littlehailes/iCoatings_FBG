import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import seaborn as sns
import os
import csv
from glob import glob
import math

Time = []
H1 = []
H2 = []
H3 = []
H4 = []
H5 = []

file_1 = "G:/My Drive/LSBU_Trip_Mar2024/Second_Substrate/03211835.txt"
file_2 = 
with open("G:/My Drive/LSBU_Trip_Mar2024/Second_Substrate/03211835.txt") as file:
    for _ in range(1):
         next(file)
    for line in file:
        columns = line.strip().split("\t")
        Time.append((columns[0]))
        H1.append((columns[2]))
        H2.append((columns[4]))
        H3.append((columns[6]))
        H4.append((columns[8]))
        H5.append((columns[10]))
        
    def root_mean_square_smoothing(data):
        smoothed_data = np.zeros_like(data, dtype=float)
        smoothed_data[0] = data[0]  # Initial value remains the same
    
        for i in range(3, len(data)):
            if abs((float(data[i])) - (float(data[i-1]))) > 5:  # Check deviation
                # Replace with average of adjacent values
                #smoothed_data[i] = (float(data[i-3]) + float(data[i-2]) + float(data[i-1])) / 3
                smoothed_data[i] = (float(data[i-3]) + float(data[i-2]) + float(data[i+3]) + float(data[i+2])) / 4
            else:
                smoothed_data[i] = data[i]
    
        return smoothed_data
    
    H1_smooth = root_mean_square_smoothing(H1) + 81.48970703 #87.00067613 Value average from stable temp values between 18:42:00-18:44:00
    H2_smooth = root_mean_square_smoothing(H2) + 47.42994023 #45.94091146
    H3_smooth = root_mean_square_smoothing(H3) + 88.53930938
    H4_smooth = root_mean_square_smoothing(H4) + 81.77456016 #83.13285
    H5_smooth = root_mean_square_smoothing(H5) + 39.00435469 #41.53197958
    
    time_axis_min = Time[0]
    time_axis_max = Time[-1]
    #fig, axs = plt.subplots(2,3, constrained_layout = True)
    fig, axs = plt.subplots( constrained_layout = True)
    #plt.subplot(2,3,1)
    #axs[0,0].plot(Time, H1_smooth)
    #axs[0,0].set_title('TC_1')
    #axs[0,0].set(xlabel = "Time", ylabel = "Temperature (oC)")
    #axs[0,0].set_ylim([0,200])
    #axs[0,0].set_xticks([Time[0],Time[-1]])
    #axs[0,0].set_xticklabels([time_axis_min, time_axis_max],rotation = 0, fontsize=9)
    #axs[0,1].plot(Time, H2_smooth)
    #axs[0,1].set_title('TC_2')
    #axs[0,1].set(xlabel = "Time", ylabel = "Temperature (oC)")
    #axs[0,1].set_ylim([0,200])
    #axs[0,1].set_xticks([Time[0],Time[-1]])
    #axs[0,1].set_xticklabels([time_axis_min, time_axis_max],rotation = 0, fontsize=9)
    #axs[0,2].plot(Time, H3_smooth)
    #axs[0,2].set_title('TC_3')
    #axs[0,2].set(xlabel = "Time", ylabel = "Temperature (oC)")
    #axs[0,2].set_ylim([0,200])
    #axs[0,2].set_xticks([Time[0],Time[-1]])
    #axs[0,2].set_xticklabels([time_axis_min, time_axis_max],rotation = 0, fontsize=9)
    #axs[1,0].plot(Time, H4_smooth)
    #axs[1,0].set_title('TC_4')
    #axs[1,0].set(xlabel = "Time", ylabel = "Temperature (oC)")
    #axs[1,0].set_ylim([0,200])
    #axs[1,0].set_xticks([Time[0],Time[-1]])
    #axs[1,0].set_xticklabels([time_axis_min, time_axis_max],rotation = 0, fontsize=9)
    #axs[1,1].plot(Time, H5_smooth)
    #axs[1,1].set_title('TC_5')
    #axs[1,1].set(xlabel = "Time", ylabel = "Temperature (oC)")
    #axs[1,1].set_ylim([0,200])
    #axs[1,1].set_xticks([Time[0],Time[-1]])
    #axs[1,1].set_xticklabels([time_axis_min, time_axis_max],rotation = 0, fontsize=9)
    # Hide the empty subplot
    #axs[1, 2].axis('off')
    
    axs.plot(Time, H5_smooth)
    axs.set_title('TC_5_Moving_Average')
    axs.set(xlabel = "Time", ylabel = "Temperature (oC)")
    axs.set_ylim([0,200])
    axs.set_xticks([Time[0],Time[-1]])
    axs.set_xticklabels([time_axis_min, time_axis_max],rotation = 0, fontsize=9)
    plt.show()
    
    #with open("Substrate2_Grating3_Temp.txt", "w") as text_file:
    #    for Time, H1_smooth in zip(Time, H1_smooth): 
    #        text_file.write(Time, H1_smooth)    
    
    def export_lists_to_text(list1, list2, filename):
        with open(filename, 'w') as file:
            for item1, item2 in zip(list1, list2):
                file.write(f"{item1},{item2}\n")



    #Output file name
    filename = 'Substrate2_S5_G1_T5_Temp.txt'
        
    # Export lists to text file
    #export_lists_to_text(Time, H5_smooth, filename)
    