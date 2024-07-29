# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 12:25:45 2024

@author: Hugh Littlehailes

Quick code to find the mean values of a noisy signal. Replaced by medFilt.
"""

import numpy as np
import csv
from glob import glob
import matplotlib.pyplot as plt
import math
import time
from datetime import datetime
from scipy.signal import find_peaks

file = "G:/My Drive/LSBU_Trip_Mar2024/Stripe5_smooth_fit.txt"

time = []
pow1 = []
pow2 = []


with open(file, 'r') as file:
    for _ in range(1):
        next(file)
    for line in file:
        columns = line.strip().split('\t')
        time.append(float(columns[0]))
        #pow1.append(float(columns[1]))
        pow2.append(float(columns[1]))

time = np.array(time)
pow2 = np.array(pow2)

peaks, _ = find_peaks(pow2)
troughs, _ = find_peaks(-pow2)

# Step 3: Calculate midpoints between adjacent peaks and troughs
midpoints_time = []
midpoints_value = []

# Combine and sort the indices of peaks and troughs
extrema_indices = np.sort(np.concatenate((peaks, troughs)))

for i in range(len(extrema_indices) - 1):
    t1, t2 = time[extrema_indices[i]], time[extrema_indices[i + 1]]
    v1, v2 = pow2[extrema_indices[i]], pow2[extrema_indices[i + 1]]
    
    # Calculate the difference between the adjacent maximum and minimum values
    if abs(v1 - v2) > 1:
        midpoint_time = (t1 + t2) / 2
        midpoint_value = (v1 + v2) / 2
        midpoints_time.append(midpoint_time)
        midpoints_value.append(midpoint_value)

# Convert to numpy arrays for easier manipulation
midpoints_time = np.array(midpoints_time)
midpoints_value = np.array(midpoints_value)

# Print the results
#print("Local maxima (peaks) indices:", peaks)
#print("Local minima (troughs) indices:", troughs)
#print("Midpoints between peaks and troughs:")
for t, v in zip(midpoints_time, midpoints_value):
    print(f"{v}")#, {v}")

# Optional: Plot the results for visualization
plt.plot(time, pow2, label='Value')
#plt.plot(time[peaks], pow2[peaks], 'ro', label='Peaks')
#plt.plot(time[troughs], pow2[troughs], 'bo', label='Troughs')
plt.plot(midpoints_time, midpoints_value, 'gx', label='Midpoints')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()
