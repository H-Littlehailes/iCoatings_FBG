# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 16:35:38 2024

@author: Hugh Littlehailes
"""
import numpy as np
import csv
from glob import glob
import matplotlib.pyplot as plt
import math
import time

plate=[0,0,25,0,0,12.5,25,12.5]
FBG1 = [12.2, 6.2]
FBG2 = [4.45, 6.3]
FBG3 = [4.35, 9.3]
FBG4 = [12.05, 8.9]
FBG5 = [20.45, 9.1]

TC1 = [4.4, 9]
TC2 = [12.5, 9.9]
TC3 = [20.5, 9.9]
TC4 = [4.6, 5.9]
TC5 = [12.4, 6.9]

#Stripe4 = [     ]
Stripe_x = [3.1, 5.8, 3.1, 5.8, 11.2, 13.9, 11.2, 13.9,19.25,21.7,19.25,21.7,3.0,5.9,3.0,5.8,11.25,13.8,11.2,13.8,19.2,21.7,19.2,21.7,]
Stripe_y = [5.75, 5.77, 3.9, 3.95, 5.6, 5.6, 3.95, 3.9, 5.65,5.6,3.95,3.9,10.2,10.2,8.3,8.25,10.1,10.05,8.15, 8.1,10,10,8.1,8.1]

#Stripe5 = [       ]
#Stripe6 = [     ]
#Stripe1 = [     ]
#Stripe2 = [     ]
#Stripe3 = [   ]
n = ['Stripe4','Stripe4','Stripe4','Stripe4','Stripe5','Stripe5','Stripe5','Stripe5','Stripe6','Stripe6','Stripe6','Stripe6',
     'Stripe1','Stripe1','Stripe1','Stripe1','Stripe2','Stripe2','Stripe2','Stripe2','Stripe3','Stripe3','Stripe3','Stripe3']

fig, ax = plt.subplots(figsize=(12.5,6.25), constrained_layout = True)
ax.plot(plate[0], plate[1], 'o', color='black')
ax.plot(plate[2], plate[3], 'o', color='black')
ax.plot(plate[4], plate[5], 'o', color='black')
ax.plot(plate[6], plate[7], 'o', color='black')

ax.plot(FBG1[0], FBG1[1], 'o', label='FBG1', color='red')
ax.plot(FBG2[0], FBG2[1], 'o', label='FBG2', color='orange')
ax.plot(FBG3[0], FBG3[1], 'o', label='FBG3', color='green')
ax.plot(FBG4[0], FBG4[1], 'o', label='FBG4', color='blue')
ax.plot(FBG5[0], FBG5[1], 'o', label='FBG5', color='magenta')
ax.plot(TC1[0], TC1[1], 'x', label='TC1', color='red')
ax.plot(TC2[0], TC2[1], 'x', label='TC2', color='orange')
ax.plot(TC3[0], TC3[1], 'x', label='TC3', color='green')
ax.plot(TC4[0], TC4[1], 'x', label='TC4', color='blue')
ax.plot(TC5[0], TC5[1], 'x', label='TC5', color='magenta')

ax.plot(Stripe_x, Stripe_y, 's', color='black')

#ax.plot(Stripe4[0], Stripe4[1], 's', color='blue')
#ax.plot(Stripe4[2], Stripe4[3], 's', color='blue')
#ax.plot(Stripe4[4], Stripe4[5], 's', color='blue')
#ax.plot(Stripe4[6], Stripe4[7], 's', color='blue', label='Stripe4')###

#ax.plot(Stripe5[0], Stripe5[1], 's', color='blue')
#ax.plot(Stripe5[2], Stripe5[3], 's', color='blue')
#ax.plot(Stripe5[4], Stripe5[5], 's', color='blue')
#ax.plot(Stripe5[6], Stripe5[7], 's', color='blue', label='Stripe5')

#ax.plot(Stripe6[0], Stripe6[1], 's', color='blue')
#ax.plot(Stripe6[2], Stripe6[3], 's', color='blue')
#ax.plot(Stripe6[4], Stripe6[5], 's', color='blue')
#ax.plot(Stripe6[6], Stripe6[7], 's', color='blue', label='Stripe6')

#ax.plot(Stripe1[0], Stripe1[1], 's', color='blue')
#ax.plot(Stripe1[2], Stripe1[3], 's', color='blue')
#ax.plot(Stripe1[4], Stripe1[5], 's', color='blue')
#ax.plot(Stripe1[6], Stripe1[7], 's', color='blue', label='Stripe1')

#ax.plot(Stripe2[0], Stripe2[1], 's', color='blue')
#ax.plot(Stripe2[2], Stripe2[3], 's', color='blue')
#ax.plot(Stripe2[4], Stripe2[5], 's', color='blue')
#ax.plot(Stripe2[6], Stripe2[7], 's', color='blue', label='Stripe2')

#ax.plot(Stripe3[0], Stripe3[1], 's', color='blue')
#ax.plot(Stripe3[2], Stripe3[3], 's', color='blue')
#ax.plot(Stripe3[4], Stripe3[5], 's', color='blue')
#ax.plot(Stripe3[6], Stripe3[7], 's', color='blue', label='Stripe3')
for i, txt in enumerate(n):
    ax.annotate(txt, (Stripe_x[i], Stripe_y[i]))
    
plt.title('Plate 2 - front')
ax.legend()
