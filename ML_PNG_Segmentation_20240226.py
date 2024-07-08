# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 11:03:15 2024

@author: Hugh Littlehailes
"""
from nptdms import TdmsFile
import os
import seaborn as sns
import tempfile
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from scipy.signal import istft, stft, butter, sosfilt, sosfreqz
import h5py

#def ML_png_seg(): # Created to subdivide the STFT PNG images to be fed into the ML model as more examples of each feedrate
import cv2
import os
    
def segment_image(image, subsection_size):
    height, width, _ = image.shape
    subsections = []
    #for y in range(0, height, subsection_size):
    #    for x in range(0, width, subsection_size):
    #                subsection = image[ y:y+subsection_size, x:x+subsection_size]
    #                subsections.append(subsection)
    for x in range(0, width, subsection_size):
                    subsection = image[ :, x:x+subsection_size]
                    subsections.append(subsection)
    return subsections
    
def export_subsections(subsections, output_dir, filename):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        for i, subsection in enumerate(subsections):
            cv2.imwrite(os.path.join(output_dir, f"{filename}_subsection_{i}.png"), subsection)
    
def process_images(input_folder, output_folder, subsection_size):
        for filename in os.listdir(input_folder):
            if filename.endswith(".png"):
                input_path = os.path.join(input_folder, filename)
                image = cv2.imread(input_path)
                subsections = segment_image(image, subsection_size)
                export_subsections(subsections, output_folder, os.path.splitext(filename)[0])
if __name__ == "__main__":
        #input_folder = "inpimages" #folder containing original png images
        #output_folder = "output_subsections" # Directory to save the subsection images to
        #subsection_size = 100 # Defines the size of each subsection -Here as 100 pixels
        
        ##process_images(input_folder, output_folder, subsection_size)
        input_folder = "G:/My Drive/LSBU_Stripe_Data/2024/Image_Based_ML/FR_Data/Coating_1/FR_15gmin"
        output_folder = "G:/My Drive/LSBU_Stripe_Data/2024/Image_Based_ML/FR_Segmented_Data/Coating_1"
        subsection_size = 50;
        process_images(input_folder, output_folder, subsection_size)