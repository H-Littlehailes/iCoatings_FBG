# -*- coding: utf-8 -*-
"""
Created on Tue May 28 11:14:00 2024

@author: Hugh Littlehailes
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import scipy
import seaborn as sns
import os
import csv
from glob import glob
import math
from scipy.signal import find_peaks
from datetime import datetime

file = "G:/My Drive/LSBU_Trip_Mar2024/P2_S5_G1_Pow2.txt"

Time = []
Pow2 = []

with open(file, 'r') as file:
    for _ in range(1):
        next(file)
    for line in file:
        columns = line.strip().split('\t')
        Time.append((columns[0]))
        Pow2.append(int(float(columns[1])))
        
# Convert Pow2 to a NumPy array
Pow2 = np.array(Pow2)
#x = np.linspace(0, len(Pow2) - 1, len(Pow2))
# Convert Time strings to total seconds
def time_to_seconds(t):
    dt = datetime.strptime(t, "%H:%M:%S.%f")
    return dt.hour * 3600 + dt.minute * 60 + dt.second + dt.microsecond / 1e6

Time_seconds = np.array([time_to_seconds(t) for t in Time])
# Find peaks (maxima)
peaks, _ = find_peaks(Pow2)
# Find troughs (minima) by finding peaks in the negative of the data
troughs, _ = find_peaks(-Pow2)

# Extract intensities
peak_intensities = Pow2[peaks]
trough_intensities = Pow2[troughs]

# Print positions and intensities
#print("Peaks (maxima):")
#for pos, intensity in zip(peaks, peak_intensities):
#    print(f"Position: {Time[pos]}, Intensity: {intensity:.3f}")

#print("\nTroughs (minima):")
#for pos, intensity in zip(troughs, trough_intensities):
#    print(f"Position: {Time[pos]}, Intensity: {intensity:.3f}")


# Write results to a text file
with open('max_min_results.txt', 'w') as f:
    f.write("Peaks (maxima):\n")
    for pos, intensity in zip(peaks, peak_intensities):
        f.write(f"{Time_seconds[pos]:.3f}, {intensity:.3f}\n")

    f.write("\nTroughs (minima):\n")
    for pos, intensity in zip(troughs, trough_intensities):
        f.write(f"{Time_seconds[pos]:.3f}, {intensity:.3f}\n")

# Fit a trendline (linear regression)
#coefficients = np.polyfit(Time_seconds, Pow2, 1)  # Fit a first-degree polynomial (a line)
#trendline = np.polyval(coefficients, Time_seconds)

# Print the equation of the trendline
#print(f"Trendline equation: y = {coefficients[0]:.3f}x + {coefficients[1]:.3f}")

#plt.figure(figsize=(10, 6))
#plt.plot(Time, Pow2, label='Data')
#plt.plot(Time[peaks], Pow2[peaks], 'ro', label='Maxima')
#plt.plot(Time[troughs], Pow2[troughs], 'bo', label='Minima')
#plt.plot(Time_seconds, trendline, 'g--', label=f'Trendline: y = {coefficients[0]:.3f}x + {coefficients[1]:.3f}')
#plt.legend()
#plt.legend()
#plt.xlabel('Time')
#plt.ylabel('Intensity')
#plt.title('Maxima and Minima in Oscillating Data')
#plt.show()
