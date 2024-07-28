# -*- coding: utf-8 -*-
"""
Created on Thu May 23 13:08:10 2024

@author: Hugh Littlehailes
"""

import numpy as np
import csv
import matplotlib.pyplot as plt
import math


P1_S3 = []
P1_S6 = []

P2_S1 = []
P2_S2 = []
P2_S3 = []
P2_S4 = []
P2_S5 = []
P2_S6 = []
P2_S1_rd = []
P2_S4_rd = []

Angle = []
file = "G:/My Drive/LSBU_Trip_Mar2024/Plate2_XRD_data.txt"
file_P1_S3 = "G:/My Drive/LSBU_Trip_Mar2024/XRD/Plate1/P1_S3_XRD.txt"
file_P1_S6 = "G:/My Drive/LSBU_Trip_Mar2024/XRD/Plate1/P1_S6_XRD.txt"
BG_data = "G:/My Drive/LSBU_Trip_Mar2024/Background_Steel_XRD.txt"
file_peaks = "G:/My Drive/LSBU_Trip_Mar2024/peaks2.txt"
P2_S1_rd = "G:/My Drive/LSBU_Trip_Mar2024/XRD/Plate2/P2_S1_Repeat_XRD.txt"
P2_S4_rd = "G:/My Drive/LSBU_Trip_Mar2024/XRD/Plate2/P2_S4_Repeat_XRD.txt"
#Ti2O_peaks = "G:/My Drive/LSBU_Trip_Mar2024/Ti2O_Peaks.txt"
#Tia_peaks = "G:/My Drive/LSBU_Trip_Mar2024/Ti-a_Peaks.txt"
Tia_peaks = "G:/My Drive/LSBU_Trip_Mar2024/Ti-alpha_Peaks.txt"
TiO_Rostoker_peaks = "G:/My Drive/LSBU_Trip_Mar2024/TiO_Rostoker1952.txt"
TiO_048 = "G:/My Drive/LSBU_Trip_Mar2024/TiO_0-48.txt"

def import_angle(file):
    Intensity = []
    with open(file, 'r') as file:
        for line in file:
            columns = line.strip().split('\t')
            Angle.append(columns[0])
    return Angle

def import_data(file, index):
    Intensity = []
    with open(file, 'r') as file:
        for line in file:
            columns = line.strip().split('\t')
            Intensity.append(float(columns[index]))
    return Intensity
Angles = import_angle(file)
BG = import_data(BG_data, index=1)

P1_S3 = import_data(file_P1_S3, index = 1)
P1_S6 = import_data(file_P1_S6, index = 1)

P2_S1 = import_data(file, index = 11)
P2_S2 = import_data(file, index = 1)
P2_S3 = import_data(file, index = 3)
P2_S4 = import_data(file, index = 5)
P2_S5 = import_data(file, index = 7)
P2_S6 = import_data(file, index = 9)
peaks = import_data(file_peaks, 1)
#Tia_peaks = import_data(Tia_peaks, 1)
#Ti2O_peaks = import_data(Ti2O_peaks, 1)

Tia_peaks = import_data(Tia_peaks, 1)
TiO_Rostoker_peaks = import_data(TiO_Rostoker_peaks, 1)
TiO_048 = import_data(TiO_048, 1)

P2_S1_redo = import_data(P2_S1_rd, index=1)
P2_S4_redo = import_data(P2_S4_rd, index=1)

## For Oxide comparison
max_s4 = max(P2_S4); P2_S4_OG = [((value / max_s4)) for value in P2_S4];
max_s1_r = max(P2_S1_redo); P2_S1_redo = [((value / max_s1_r)+1) for value in P2_S1_redo];
max_s4_r = max(P2_S4_redo); P2_S4_redo = [((value / max_s4_r)+1) for value in P2_S4_redo];

max_p1_s3_r = max(P1_S3); P1_S3 = [((value / max_p1_s3_r)) for value in P1_S3];
max_p1_s6_r = max(P1_S6); P1_S6 = [((value / max_p1_s6_r)+1) for value in P1_S6];

max_BG = max(BG); BG = [value / max_BG for value in BG];
max_s6 = max(P2_S6); P2_S6 = [((value / max_s6)+5) for value in P2_S6];
max_s1 = max(P2_S1); P2_S1 = [value / max_s6 for value in P2_S1];
max_s2 = max(P2_S2); P2_S2 = [((value / max_s6)+1) for value in P2_S2];
max_s3 = max(P2_S3); P2_S3 = [((value / max_s6)+2) for value in P2_S3];
max_s4 = max(P2_S4); P2_S4 = [((value / max_s6)+3) for value in P2_S4];
max_s5 = max(P2_S5); P2_S5 = [((value / max_s6)+4) for value in P2_S5];


maxmax = max([max_s1_r, max_s2, max_s3, max_s4_r, max_s5, max_s6]);
print(maxmax)
## Values for main plot 
P2_S1_redo1 = [((value)-1) for value in P2_S1_redo];
P2_S4_redo1 = [((value)+2) for value in P2_S4_redo];

#key_x_values = [ 35.0847745, 38.43041381, 40.16915529, 53.01068215, 74.14463274]

axes_label = [20, 30, 40, 50, 60, 70, 80]
axes_position = [Angle[0], Angle[381], Angle[762], Angle[1143], Angle[1524], Angle[1904], Angle[-1]]

Ti_y = [6,6,6,6,6]
    

max_p = max(P2_S6)
min_p = min(P2_S1)


fig0,ax1 = plt.subplots(1,1,constrained_layout=True) 
#ax1.plot(Angles, BG)
ax1.plot(Angles, P2_S1,color ='k')
ax1.plot(Angles, P2_S2,color ='k')
ax1.plot(Angles, P2_S3,color ='k')
ax1.plot(Angles, P2_S4,color ='k')
ax1.plot(Angles, P2_S5,color ='k')
ax1.plot(Angles, P2_S6,color ='k')
#ax1.plot(Angles, BG)
ax1.plot(Angles, Tia_peaks,color ='k',linewidth=0.8 )
#ax1.plot(Angles, Ti2O_peaks,color ='m',linewidth=0.8 )
ax1.plot(Angles, TiO_Rostoker_peaks,color ='m',linewidth=0.8 )
ax1.plot(Angles, TiO_048,color ='g',linewidth=0.8 )
ax1.set(xlabel="Angle (2$\\theta $)",ylabel="Intensity [a.u.]")
ax1.set_ylim([0,6])
plt.xticks(axes_position, axes_label)
plt.yticks([])

axes_label1 = [20, 30, 40, 50, 60, 70, 80]
axes_position1 = [Angle[0], Angle[381], Angle[762], Angle[1143], Angle[1524], Angle[1904], Angle[-1]]
fig1, (ax1, ax2) = plt.subplots(1,2, constrained_layout = True)
ax1.plot(Angles, P2_S1)
ax1.plot(Angles, P2_S1_redo)
ax1.plot(Angles, peaks,color ='k',linewidth=0.8 )
ax1.set(xlabel="Angle (2$\\theta $)",ylabel="Intensity [a.u.]", title = "Plate 2 Sample 1 - 535 A")
ax1.set_xticks(axes_position1, axes_label1)
ax1.set_ylim([0,2])
ax2.plot(Angles, P2_S4_OG)
ax2.plot(Angles, P2_S4_redo)
ax2.plot(Angles, peaks,color ='k',linewidth=0.8 )
ax2.set(xlabel="Angle (2$\\theta $)",ylabel="Intensity [a.u.]", title = "Plate 2 Sample 4 - 545 A")
ax2.set_xticks(axes_position1, axes_label1)
ax2.set_ylim([0,2])
#ax2.xticks(axes_position, axes_label)

fig2,ax1 = plt.subplots(1,1,constrained_layout=True) 
#ax1.plot(Angles, BG)
ax1.plot(Angles, P1_S3,color ='k')
ax1.plot(Angles, P1_S6,color ='k' )
#ax1.plot(Angles, peaks,color ='k',linewidth=0.8 )
ax1.plot(Angles, Tia_peaks,color ='k',linewidth=0.8 )
#ax1.plot(Angles, Ti2O_peaks,color ='m',linewidth=0.8 )
ax1.plot(Angles, TiO_Rostoker_peaks,color ='m',linewidth=0.8 )
ax1.plot(Angles, TiO_048,color ='g',linewidth=0.8 )
ax1.set(xlabel="Angle (2$\\theta $)",ylabel="Intensity [a.u.]")
ax1.set_ylim([0,2])
plt.xticks(axes_position, axes_label)
plt.yticks([])
