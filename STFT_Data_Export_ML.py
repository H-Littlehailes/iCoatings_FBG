# -*- coding: utf-8 -*-
"""
Created on Tue Jan 30 17:11:47 2024

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

# time periods where the stripes occur in each file
CLIP_PERIOD = {'sheff_lsbu_stripe_coating_1': [120.0, 180.0],  # 180
               'sheff_lsbu_stripe_coating_2': [1.0, 350.0],  # [100.0, 350.0],
               # 'sheff_lsbu_stripe_coating_2_ref' : [1.0,99.0],
               'sheff_lsbu_stripe_coating_3_pulsing': [100.0, 180.0]}

# stripe locations 0-1000
STRIPE_PERIOD = {'sheff_lsbu_stripe_coating_1': {#'stripe_1': [120.2635, 125.2635],
                                                 'stripe_2': [128.375, 129.684],
                                                 'stripe_3': [141.875, 143.184],
                                                 'stripe_4': [152.015, 153.324],
                                                 'stripe_5': [163.485, 164.794],
                                                 'stripe_6': [175.925, 177.234]
                                                 }}
# first pass
STRIPE_PERIOD_2 = {'sheff_lsbu_stripe_coating_2': {#'ref':  [281, 281.5],#[101,106], 
                                                   'stripe_1_1': [108.430, 109.083], #[107.800, 109.060], 
                                                   'stripe_2_1': [119.330, 119.983], #[118.700, 119.960],
                                                   'stripe_3_1': [129.230, 129.883], #[128.600, 129.860]
                                                   'stripe_4_1': [138.690, 139.343], #[138.060, 139.320]
                                                   'stripe_5_1': [148.245, 148.898], #[147.615, 148.875]
                                                   # 2nd pass
                                                   #'stripe_1_2': [158.830, 159.483], #[158.2, 159.37]
                                                   #'stripe_2_2': [170.830, 171.483], #[170.2, 171,46]
                                                   #'stripe_3_2': [181.800, 182.453], #[181.17, 182.43]
                                                   #'stripe_4_2': [191.580, 192.233], #[190.95, 192.21]
                                                   #'stripe_5_2': [201.230, 201.883], #[200.6, 201.86]
                                                   # 3rd pass
                                                   #'stripe_1_3': [216.510, 217.163], #[215.88, 217.14]
                                                   #'stripe_2_3': [227.880, 228.533], #[227.25, 228.51]
                                                   #'stripe_3_3': [238.890, 239.543], #[238.26, 239.52]
                                                   #'stripe_4_3': [251.160, 251.813], #[250.53, 251.79]
                                                   #'stripe_5_3': [262.120, 262.773], #[261.49, 262.75]
                                                   # unknown
                                                   #'stripe_1_4': [278.350, 279.003], #[277.72, 278.98]
                                                   #'stripe_2_4': [292.210, 292.863], #[291.58, 292.84]
                                                   #'stripe_3_4': [306.050, 306.703], #[305.42, 306.68]
                                                   #'stripe_4_4': [316.380, 317.033], #[315.75, 317.01]
                                                   #'stripe_5_4': [328.91, 329.564], #[328.28, 329.54]
                                                   }}

#Full clipped data                                 {#'ref':  [281, 281.5],#[101,106], 
                                                   #'stripe_1_1': [107.775, 109.084], #[107.800, 109.060], 
                                                   #'stripe_2_1': [118.675, 119.984], #[118.700, 119.960],
                                                   #'stripe_3_1': [128.575, 129.884], #[128.600, 129.860]
                                                   #'stripe_4_1': [138.035, 139.344], #[138.060, 139.320]
                                                   #'stripe_5_1': [147.590, 148.899], #[147.615, 148.875]
                                                   ## 2nd pass
                                                   #'stripe_1_2': [158.175, 159.484], #[158.2, 159.37]
                                                   #'stripe_2_2': [170.175, 171.484], #[170.2, 171,46]
                                                   #'stripe_3_2': [181.145, 182.454], #[181.17, 182.43]
                                                   #'stripe_4_2': [190.925, 192.234], #[190.95, 192.21]
                                                   #'stripe_5_2': [200.575, 201.884], #[200.6, 201.86]
                                                   ## 3rd pass
                                                   #'stripe_1_3': [215.855, 217.164], #[215.88, 217.14]
                                                   #'stripe_2_3': [227.225, 228.534], #[227.25, 228.51]
                                                   #'stripe_3_3': [238.235, 239.544], #[238.26, 239.52]
                                                   #'stripe_4_3': [250.505, 251.814], #[250.53, 251.79]
                                                   #'stripe_5_3': [261.465, 262.774], #[261.49, 262.75]
                                                   ## unknown
                                                   #'stripe_1_4': [277.695, 279.004], #[277.72, 278.98]
                                                   #'stripe_2_4': [291.555, 292.864], #[291.58, 292.84]
                                                   #'stripe_3_4': [305.395, 306.704], #[305.42, 306.68]
                                                   #'stripe_4_4': [315.725, 317.034], #[315.75, 317.01]
                                                   #'stripe_5_4': [328.255, 329.564], #[328.28, 329.54]
                                                   #}}

STRIPE_PERIOD_3 = {'sheff_lsbu_stripe_coating_3_pulsing': {#'stripe_1': [105, 110],
                                                           #'stripe_2': [112, 117],
                                                           #'stripe_3': [119, 124],
                                                           #'stripe_4': [125, 131],
                                                           #'stripe_5': [134, 139],
                                                           #'stripe_6': [141, 147],
                                                           #'stripe_7': [148, 154],
                                                           #'stripe_8': [157, 163],
                                                           'stripe_9': [164, 169], #[164, 170],
                                                           #'stripe_10': [172, 178],
                                                           }}
PERIODS = [STRIPE_PERIOD, STRIPE_PERIOD_2, STRIPE_PERIOD_3]

# dict for mapping stripe number to feed rate
FEED_RATE_1 = {'sheff_lsbu_stripe_coating_1':   {#'stripe_1': '15 G/MIN',
                                                 'stripe_2': '15 G/MIN',
                                                 'stripe_3': '20 G/MIN',
                                                 'stripe_4': '25 G/MIN',
                                                 'stripe_5': '30 G/MIN',
                                                 'stripe_6': '35 G/MIN'
                                                 }}

FEED_RATE_2 = {'sheff_lsbu_stripe_coating_2': {#'ref'       : 'ref',
                                               'stripe_1_1': '15 G/MIN',
                                               'stripe_2_1': '20 G/MIN',
                                               'stripe_3_1': '25 G/MIN',
                                               'stripe_4_1': '30 G/MIN',
                                               'stripe_5_1': '35 G/MIN',
                                               # 2nd pass
                                               #'stripe_1_2': '15 G/MIN (2)',
                                               #'stripe_2_2': '20 G/MIN (2)',
                                               #'stripe_3_2': '25 G/MIN (2)',
                                               #'stripe_4_2': '30 G/MIN (2)',
                                               #'stripe_5_2': '35 G/MIN (2)',
                                               # 3rd pass
                                               #'stripe_1_3': '15 G/MIN (3)',
                                               #'stripe_2_3': '20 G/MIN (3)',
                                               #'stripe_3_3': '25 G/MIN (3)',
                                               #'stripe_4_3': '30 G/MIN (3)',
                                               #'stripe_5_3': '35 G/MIN (3)',
                                               # unknown
                                               #'stripe_1_4': '15 G/MIN (4)',
                                               #'stripe_2_4': '20 G/MIN (4)',
                                               #'stripe_3_4': '25 G/MIN (4)',
                                               #'stripe_4_4': '30 G/MIN (4)',
                                               #'stripe_5_4': '35 G/MIN (4)',
                                               }}

FEED_RATE = [FEED_RATE_1, FEED_RATE_2]


# from https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
# band pass
def butter_bandpass(lowcut, highcut, order=5):
    nyq = 0.5 * 1e6
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='bandpass', analog=False, output='sos')


def butter_bandpass_filter(data, lowcut, highcut, order=5):
    sos = butter_bandpass(lowcut, highcut, 1e6, order=5)
    return sosfilt(sos, data)

# band stop


def butter_bandstop(lowcut, highcut, order=5):
    nyq = 0.5 * 1e6
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='bandstop', analog=False, output='sos')


def butter_bandstop_filter(data, lowcut, highcut, order=5):
    sos = butter_bandstop(lowcut, highcut, 1e6, order=order)
    return sosfilt(sos, data)

# general band filter


def butter_band(lowcut, highcut, btype, order=5):
    nyq = 0.5 * 1e6
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], analog=False, output='sos')


def butter_bfilter(data, lowcut, highcut, btype, order=5):
    sos = butter_band(lowcut, highcut, 1e6, btype, order)
    return sosfilt(sos, data)


def applyFilters(data, freq, mode, **kwargs):
    '''
        Apply a series of filters to the given daa

        freq is either a single cutoff frequency or a 2-element collection for band filters
        mode is either a single string or list of strings if multiple filters are specified in mode.

        Inputs:
            data : Array of values
            freq : Single value or list of frequencies
            mode : String or list of strings defining type of filters

        Returns filtered data
    '''
    # ensure frequencies and modes are lists
    if isinstance(freq, (int, float)):
        freq = [freq,]
    # if mode is a single string them treat all filters as that mode
    if isinstance(mode, str):
        modes = len(freq)*[mode,]
    elif len(freq) != len(mode):
        raise ValueError("Number of modes must match the number of filters!")
    else:
        modes = list(mode)
    # iterate over each filter and mode
    for c, m in zip(freq, modes):
        # if it's a single value then it's a highpass/lowpass filter
        if isinstance(c, (int, float)):
            sos = butter(kwargs.get("order", 10), c/1e6, m,
                         fs=1e6, output='sos', analog=False)
            data = sosfilt(sos, data)
        # if it's a list/tuple then it's a bandpass filter
        elif isinstance(c, (tuple, list)):
            if m == "bandpass":
                data = butter_bandpass_filter(
                    data, c[0], c[1], kwargs.get("order", 10))
            elif m == "bandstop":
                data = butter_bandpass_filter(
                    data, c[0], c[1], kwargs.get("order", 10))
    return data


def plotFreqResponse(freq, btype, pts=int(1e6/4), **kwargs):
    '''
        Butterworth filter frequency response

        freq is either a single cutoff frequency or a 2-element collection for band filters
        pts is the number of points used for plotting. Passed to worN parameter in sosfreqz.

        Inputs:
            freq : Single or 2-element collection of cutoff frequencies
            btype : String for frequency type. See https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html.
            pts : Number of points for plotting. Default 1e6/4
            order : Filter model order

        Returns figure
    '''
    # digital filters require freq be between 0 and 1
    if isinstance(freq, (float, int)):
        freq_norm = freq/(1e6/2)
    else:
        freq_norm = freq
    # if low or highpass
    if btype in ["lowpass", "highpass"]:
        sos = butter(kwargs.get("order", 10), freq_norm,
                     btype, analog=False, output='sos')
    # for band filters
    elif btype == "bandpass":
        sos = butter_bandpass(freq[0], freq[1], 1e6, kwargs.get("order", 10))
    elif btype == "bandstop":
        sos = butter_bandstop(freq[0], freq[1], 1e6, kwargs.get("order", 10))
    # generate freq response
    w, h = sosfreqz(sos, worN=pts, fs=1e6)
    f, ax = plt.subplots(nrows=2, constrained_layout=True)
    ax[0].plot(w, np.abs(h))
    ax[1].plot(w, np.angle(h))
    ax[0].set(xlabel="Frequency (Hz)", ylabel="Gain")
    ax[1].set(xlabel="Frequency (Hz)", ylabel="Phase (radians)")
    ax[0].vlines(freq, 0, 1, color='purple')
    ax[1].vlines(freq, -2*np.pi, 2*np.pi, color='purple')
    f.suptitle("Butterworth Frequency Response")
    return f


def loadTDMSData(path):
    '''
        Load in the TDMS data and add a column for time

        All the functions were built around there being a Time columns rather than using index as time

        Columns are renamed from their full paths to Input 0 and Input 1

        Returns pandas dataframe
    '''
    data = TdmsFile(path).as_dataframe(time_index=False)
    data['Time (s)'] = np.arange(data.shape[0])/1e6
    data.rename(columns={c: c.split('/')[-1].strip("'")
                for c in data.columns}, inplace=True)
    return data


def loadTDMSSub(path, chunk, is_time=False):
    '''
        Load in a sub-section of TDMS data

        Helps avoid loading in the entire file

    '''
    with TdmsFile(path) as tdms_file:
        # find group that contains recording
        group = list(
            filter(lambda x: "Recording" in x.name, tdms_file.groups()))[0]
        nf = group['Input 0'].data.shape[0]
        mt = nf*1e-6
        # convert to index
        if is_time:
            chunk = [max(0, min(nf, int(c*1e6))) for c in chunk]
        nf = group['Input 0'][chunk[0]:chunk[1]].shape[0]
        time = np.arange(nf)*1e-6
        time += chunk[0]*1e-6
        return time, group['Input 0'][chunk[0]:chunk[1]], group['Input 1'][chunk[0]:chunk[1]]


def PolyArea(x, y):
    return 0.5*np.abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1)))


def filterStripesCalcArea(fn, freq=50e3, mode='highpass', dist=int(50e3), **kwargs):
    from scipy.signal import find_peaks
    import matplotlib.patches as mpatches
    from matplotlib.pyplot import cm
    sns.set_theme("paper")
    fname = os.path.splitext(os.path.basename(fn))[0]
    # check if supported
    periods = list(filter(lambda x: fname in x, PERIODS))
    feedrate = list(filter(lambda x: fname in x, FEED_RATE))
    if periods:
        periods = periods[0][fname]
        # ensure frequencies and modes are lists
        if isinstance(freq, (int, float)):
            freq = [freq,]
        # if mode is a single string them treat all filters as that mode
        if isinstance(mode, str):
            modes = len(freq)*[mode,]
        elif len(freq) != len(mode):
            raise ValueError(
                "Number of modes must match the number of filters!")
        else:
            modes = list(mode)
        # format frequencies and modes into strings for plotting titles and saving
        modes_save_string = '-'.join(modes)
        modes_plot_string = ','.join(modes)
        freq_save_string = '-'.join([str(c) for c in freq])
        freq_plot_string = ','.join([str(c) for c in freq])
        # filter to time periods dict
        for sname, chunk in periods.items():
            # load sub bit of the stripe
            time, i0, i1 = loadTDMSSub(fn, chunk, is_time=True)
            # plot period
           # f_i0,ax_i0 = plt.subplots(constrained_layout=True)
          #  ax_i0.plot(time,i0,'b-',label="Original")
            # plot period
          #  f_i1,ax_i1 = plt.subplots(constrained_layout=True)
          #  ax_i1.plot(time,i0,'b-',label="Original")
            # iterate over each filter and mode
            for c, m in zip(freq, modes):
                print(sname, c, m)
                # if it's a single value then it's a highpass/lowpass filter
                if isinstance(c, (int, float)):
                    print("bing!")
                    sos = butter(kwargs.get("order", 10), c, m,
                                 fs=1e6, output='sos', analog=False)
                    i0 = sosfilt(sos, i0)
                    i1 = sosfilt(sos, i1)
                # if it's a list/tuple then it's a bandpass filter
                elif isinstance(c, (tuple, list)):
                    print("bong!")
                    if m == "bandpass":
                        i0 = butter_bandpass_filter(
                            i0, c[0], c[1], kwargs.get("order", 10))
                        i1 = butter_bandpass_filter(
                            i1, c[0], c[1], kwargs.get("order", 10))
                    elif m == "bandstop":
                        i0 = butter_bandstop_filter(
                            i0, c[0], c[1], kwargs.get("order", 10))
                        i1 = butter_bandstop_filter(
                            i1, c[0], c[1], kwargs.get("order", 10))
            #print(max(i1, key=abs))
            # plot data
   #         ax_i0.plot(time,i0,'r-',label="Filtered")
    #        ax_i0.legend()
     #       ax_i0.set(xlabel="Time (s)",ylabel="Voltage (V)",title=f"{fname}, Input 0, {sname}, {modes_plot_string}, {freq_plot_string}Hz")
      #      f_i0.savefig(fr"{os.path.splitext(fn)[0]}-Input 0-time-stripe-{sname}-filtered-{modes_save_string}-freq-{freq_save_string}.png")
       #     plt.close(f_i0)
        #    #plot data
         #   ax_i1.plot(time,i1,'r-',label="Filtered")
          #  ax_i1.legend()
           # ax_i1.set(xlabel="Time (s)",ylabel="Voltage (V)",title=f"{fname}, Input 1, {sname}, {modes_plot_string}, {freq_plot_string}Hz")
            #f_i1.savefig(fr"{os.path.splitext(fn)[0]}-Input 1-time-stripe-{sname}-filtered-{modes_save_string}-freq-{freq_save_string}.png")
            # plt.close(f_i1)
    # else:
     #   print(f"Unsupported file {fn}!")


# def calcStripeAreas(path,dist=int(50e3),**kwargs):

    # check if supported
   # periods = list(filter(lambda x : os.path.splitext(os.path.basename(fn))[0] in x,PERIODS))
  #  feedrate = list(filter(lambda x : os.path.splitext(os.path.basename(fn))[0] in x,FEED_RATE))
    if feedrate:
        # Select Feedrate base on the source file
        feedrate = feedrate[0][fname]
    if periods:
        periods = periods  # [0][fname]

        color = cm.rainbow(np.linspace(0, 1, len(periods)))
        time_pts_i0 = []
        v_pts_i0 = []
        area_i0 = []

        time_pts_i1 = []
        v_pts_i1 = []
        area_i1 = []
        # filter to time periods dict
        for (sname, chunk), c in zip(periods.items(), color):
           # time,i0,i1 = loadTDMSSub(fn,chunk,is_time=True)
            # find edge of Input 0
            # mask signal to +ve
            mask_filt = i0.copy()
            mask_filt[i0 < 0] = 0
            # find peaks in the signal
            pks = find_peaks(mask_filt, distance=dist)[0]
            if len(pks) == 0:
                raise ValueError(f"Failed to find +ve edge in Input 0!")

            time_pts_i0.extend(time[pks].tolist())
            v_pts_i0.extend(i0[pks].tolist())

            # mask signal to -ve
            mask_filt = i0.copy()
            mask_filt[i0 > 0] = 0
            mask_abs = np.abs(mask_filt)
            # find peaks in the signal
            pks = find_peaks(mask_abs, distance=dist)[0]
            if len(pks) == 0:
                raise ValueError(f"Failed to find -ve edge in Input 0!")

            # add the points so that it's in clockwise order
            time_pts_i0.extend(time[pks].tolist()[::-1])
            v_pts_i0.extend(i0[pks].tolist()[::-1])
            area_i0.append(PolyArea(time_pts_i0, v_pts_i0))

           # f0,ax0 = plt.subplots(constrained_layout=True,figsize=(9,8))
            #plt.plot(time_pts_i0,v_pts_i0, 'r-')
            #ax0.set_title(f"{os.path.splitext(os.path.basename(path))[0]}, Input 0")

            # find edge of Input 1
            # mask signal to +ve
            mask_filt = i1.copy()
            mask_filt[i1 < 0] = 0
            # find peaks in the signal
            pks = find_peaks(mask_filt, distance=dist)[0]
            if len(pks) == 0:
                raise ValueError(f"Failed to find +ve edge in Input 1!")

            time_pts_i1.extend(time[pks].tolist())
            v_pts_i1.extend(i0[pks].tolist())

            # mask signal to -ve
            mask_filt = i1.copy()
            mask_filt[i1 > 0] = 0
            mask_abs = np.abs(mask_filt)
            # find peaks in the signal
            pks = find_peaks(mask_abs, distance=dist)[0]
            if len(pks) == 0:
                raise ValueError(f"Failed to find -ve edge in Input 1!")

            time_pts_i1.extend(time[pks].tolist()[::-1])
            v_pts_i1.extend(i0[pks].tolist()[::-1])
            area_i1.append(PolyArea(time_pts_i1, v_pts_i1))

            #f1,ax1 = plt.subplots(constrained_layout=True,figsize=(9,8))
            #plt.plot(time_pts_i1,v_pts_i1, 'b--')
            #ax1.set_title(f"{os.path.splitext(os.path.basename(path))[0]}, Input 1")
            # return f0,f1

            time_pts_i0.clear()
            v_pts_i0.clear()

            time_pts_i1.clear()
            v_pts_i1.clear()
        # plot
        f, ax = plt.subplots(ncols=2, constrained_layout=True)
        ax[0].plot(area_i0)
        #ax[0].set(xlabel="Stripe Index",ylabel="Area (V*t)",title="Input 0")
        ax[0].set_xlabel("Stripe Index", fontsize=24)
        ax[0].set_ylabel("Area (V*t)", fontsize=24)
        ax[0].set_title("Input 0", fontsize=24)
        ax[1].plot(area_i1)
        #ax[1].set(xlabel="Stripe Index",ylabel="Area (V*t)",title="Input 1")
        ax[1].set_xlabel("Stripe Index", fontsize=24)
        ax[1].set_ylabel("Area (V*t)", fontsize=24)
        ax[1].set_title("Input 1", fontsize=24)
        #ax.tick_params(axis='both', labelsize=22)
        #f.suptitle(os.path.splitext(os.path.basename(path))[0]+" Shoelace Area between Edges")
        return f


# (fn,freq=50e3,mode='highpass',**kwargs)
def bandpassfilterStripes(fn, freq=[100e3, 250e3], mode=['highpass', 'lowpass'], **kwargs):
    '''
        Apply filters to each identified stripe in the file

        A filter can be applied to the signal using the mode, order and and cutoff_freq keywords.
        The cutoff_freq keyword is the cutoff frequency of the filter.
        Can either be:
            - Single value representing a highpass/lowpass filter
            - 2-element Tuple/list for a bandpass filter.
            - List of values or list/tuples for a series of filters applied sequentially

        The mode parameter is to specify whether the filters are lowpass ("lp") or highpass ("hp"). If it's a single string, then it's applied
        to all non-bandpass filters in cutoff_freq. If it's a list, then it MUST be the same length as the number of filters

        Generated plots are saved in the same location as the source file

        This is intended to remove the noise floor.

        Inputs:
            fn : TDMS path
            freq : Cutoff freq. Float, list of floats or list of tuples/lists. Default 50e3.
            order : Filter order. Default 10.
    '''
    from scipy import signal
    fname = os.path.splitext(os.path.basename(fn))[0]

    # check if supported
    #periods = list(filter(lambda x : os.path.splitext(os.path.basename(path))[0] in x,PERIODS))
    #feedrate = list(filter(lambda x : os.path.splitext(os.path.basename(path))[0] in x,FEED_RATE))
    # if feedrate:
    #    feedrate = feedrate[0][os.path.splitext(os.path.basename(path))[0]] #Select Feedrate base on the source file
    # if periods:
    #   periods = periods[0][os.path.splitext(os.path.basename(path))[0]]
    periods = list(filter(lambda x: fname in x, PERIODS))
    #print(periods)
   # ref = list(filter(lambda y: 'ref' in y, PERIODS))

    freq = [100e3, 250e3]
    if periods:

       # periods2 = periods[0][fname1]
        periods = periods[0][fname]
        #print(periods)
        # ensure frequencies and modes are lists
        if isinstance(freq, (int, float)):
            freq = [freq,]
        # if mode is a single string them treat all filters as that mode
        if isinstance(mode, str):
            modes = len(freq)*[mode,]
        elif len(freq) != len(mode):
            raise ValueError(
                "Number of modes must match the number of filters!")
        else:
            modes = list(mode)
        # format frequencies and modes into strings for plotting titles and saving
        modes_save_string = '-'.join(modes)
        modes_plot_string = ','.join(modes)
        freq_save_string = '-'.join([str(c) for c in freq])
        freq_plot_string = ','.join([str(c) for c in freq])
        # filter to time periods dict
        # for sname,chunk in periods2.items():
        #   time1,iref0,iref1 = loadTDMSSub(fn1,chunk,is_time=True)

        #print(periods.items())
       # def get_nth_key(periods,n=0):
       #     if n<0:
       #         n+=len(periods)
      #      for i, key in enumerate(periods.keys()):
       #         if i==n:
      #              return key
       # key = get_nth_key(periods, n=1)
       # print(key)
        n_time = [] # Empty list for time
        n_i0 = []   # Empty list for input 0 values
        n_i1 = []   # Empty list for input 1 values
        ai0 = []    # Empty list for noise subtracted input 0 values
        ai1 = []    # Empty list for noise subtracted input 1 values
        ri0 = []
        ri1 = []
        
        # Loop through all listed stripe chunks and output the respective values to the
        # respective empty lists.
        for x in periods:
            for sname, chunk in periods.items(): # stripe name and chunk
                # load sub bit of the stripe
                time, i0, i1 = loadTDMSSub(fn, chunk, is_time=True)
                n_time.append(time)
                n_i0.append(i0)
                n_i1.append(i1)
                #print(len(n_time))
                #print(chunk)
                #time1,iref0,iref1 = loadTDMSSub(fn,chunk,is_time=True)
           # print(len(n_i0[1]))
            print(len(n_i1))
                # plot period
            
            #for x in n_i0:
           # bi0, ri0 = signal.deconvolve(n_i0[1], n_i0[0])    
            
           # ai0 = np.append(ai0, bi0) #ai0 = ai0.append(ai0)
            #ri0 = np.append(ri0, ri0) #ri0 = ri0.append(ri0)
            #for x in n_i1:
            #ai1, ri1 = signal.deconvolve(n_i1[1], n_i1[0])
            #ai1 = np.append(ai1, ai1) #ai1 = ai1.append(ai1)
            #ri1 = np.append(ri1, ri1) #ri1 = ri1.append(ri1)
          #  print(len(ai0))
         #   print(len(ri0))
          #  print(len(n_time))
            
            ############## Mean subtraction ###################################                
            #def mean_positive(L):
            #    # Get all positive numbers into another list
            #    pos_only = [x for x in L if np.any(L) > 0]
            #    if pos_only:
            #        return sum(pos_only) /  len(pos_only)
            #    raise ValueError('No postive numbers in input')
            #def mean_negative(L):
            #    # Get all negative numbers into another list
            #    neg_only = [x for x in L if np.any(L) < 0]
            #    if neg_only:
            #        return sum(neg_only) /  len(neg_only)
            #    raise ValueError('No negative numbers in input')
# 
            #print('bong')
            
            #positiveAvgRef = mean_positive(n_i0[0])
            #negativeAvgRef = mean_negative(n_i0[0])
            
            #def BackgroundRemovei0(M):
            #    for x in M:
            #        if np.any(M) > 0:
            #        #if x >= 0 :
            #            t = x - mean_positive(n_i0[0])
            #        else :
            #            t = x - mean_negative(n_i0[0])
            #    return(t)
           # 
            #def BackgroundRemovei1(M):
            #    for x in M:
            #        if np.any(M) > 0:
            #        #if x >= 0 :
            #            u = x - mean_positive(n_i1[0])
            #        else :
            #            u = x - mean_negative(n_i1[0])
            #    return(u)
            #print("ding!")
            #ai0 = ai0.append(BackgroundRemovei0(n_i0[1]))
            #print("dong!")
            #ai1 = ai1.append(BackgroundRemovei1(n_i1[1]))
            ############## Mean subtraction ###################################
            
            #print(mean_positive(n_i0[0]))
            #print(mean_negative(n_i0[0]))
            #for y in n_i0:
            #    n = n_i0[1]-n_i0[0]
            #    ai0.append(n)

            #for y in n_i1:
            #    u = n_i1[1]-n_i1[0]
            #    ai1.append(u)
            
            #for i in n_i0:
            #    ai0.append(n_i0[1] - n_i0[0])
           # print(len(ai0))
            #print(len(n_time))
            #for l in n_i1:
            #    ai1.append(n_i1[1] - n_i1[0])
            
            #ai0 = np.array(n_i0[1]) - np.array(n_i0[0])
           # ai1 = np.array(n_i1[1]) - np.array(n_i1[0])
                
            ai0 = n_i0[1] - n_i0[0]
            ai1 = n_i1[1] - n_i1[0]
            
            f_i0, ax_i0 = plt.subplots(constrained_layout=True)
            ax_i0.plot(n_time[1], n_i0[1], 'b-', label="Original")
            ax_i0.plot(n_time[1], n_i0[0], 'g-', label="Background")
            ax_i0.plot(n_time[1], ai0, 'k-', label="Background removed")
                    # plot period
            f_i1, ax_i1 = plt.subplots(constrained_layout=True)
            ax_i1.plot(n_time[1], n_i1[1], 'b-', label="Original")
            ax_i1.plot(n_time[1], n_i1[0], 'g-', label="Background")
            ax_i1.plot(n_time[1], ai1, 'k-', label="Background removed")
                # iterate over each filter and mode
                # for c,m in zip(freq,modes):
                #   print(sname,c,m)
                # if it's a single value then it's a highpass/lowpass filter
                #   if isinstance(c,(int,list)):
            print("bing!")

               # i0 = i0 - iref0
               #i1 = i1 - iref1

            sos = butter(kwargs.get("order", 10), 100e3, 'highpass',
                         fs=1e6, output='sos', analog=False)
            i0 = sosfilt(sos, ai0)
            i1 = sosfilt(sos, ai1)
            sos = butter(kwargs.get("order", 10), 250e3, 'lowpass',
                         fs=1e6, output='sos', analog=False)
            i0 = sosfilt(sos, i0)
            i1 = sosfilt(sos, i1)
            # if it's a list/tuple then it's a bandpass filter
            #  elif isinstance(c,(tuple,list)):
            #    print("bong!")
            #     if m == "bandpass":
            #        i0 = butter_bandpass_filter(i0,c[0],c[1],kwargs.get("order",10))
            ##        i1 = butter_bandpass_filter(i1,c[0],c[1],kwargs.get("order",10))
            #    elif m == "bandstop":
            #        i0 = butter_bandstop_filter(i0,c[0],c[1],kwargs.get("order",10))
            #       i1 = butter_bandstop_filter(i1,c[0],c[1],kwargs.get("order",10))
           #print(max(i1, key=abs))
           # plot data
            ax_i0.plot(n_time[1], i0, 'r-', label="Filtered")
            ax_i0.legend()
            # ,title=f"{fname}, Input 0, {sname}, {modes_plot_string}, {freq_plot_string}Hz", fontsize=24)
            ax_i0.set(xlabel="Time (s)", ylabel="Voltage (V)")
            ax_i0.set_title(
                f"{fname}, Input 0, {sname}, {modes_plot_string}, {freq_plot_string}Hz", fontsize=9)
            #f_i0.savefig(fr"{os.path.splitext(fn)[0]}-Input 0-time-stripe-{sname}-filtered-{modes_save_string}-freq-{freq_save_string}.png")
            # plt.close(f_i0)
            # plot data

            ax_i1.plot(n_time[1], i1, 'r-', label="Filtered")
            ax_i1.legend()
            # ,title=f"{fname}, Input 1, {sname}, {modes_plot_string}, {freq_plot_string}Hz")
            ax_i1.set(xlabel="Time (s)", ylabel="Voltage (V)")
            ax_i1.set_title(
                f"{fname}, Input 1, {sname}, {modes_plot_string}, {freq_plot_string}Hz", fontsize=9)
            #f_i1.savefig(fr"{os.path.splitext(fn)[0]}-Input 1-time-stripe-{sname}-filtered-{modes_save_string}-freq-{freq_save_string}.png")
            # plt.close(f_i1)

            return f_i0, f_i1
    else:
        print(f"Unsupported file {fn}!")



def stripeSpecgram(fn,shift=False):
    '''
        Plot the spectrogram of each stripe in the target file

        Each file is 

        Inputs:
            fn : TDMS path
    '''
    from scipy import signal
    #from scipy.signal import spectrogram
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    #from matplotlib.colors import LogNorm
    fname = os.path.splitext(os.path.basename(fn))[0]
    # check if supported
    Zxx_i0 = []
    Zxx_i1 = []
    Zxx_newi0 = []
    Zxx_newi1 = []
    t = []
    f = []
    
    periods = list(filter(lambda x : fname in x,PERIODS))
    if periods:
        periods = periods[0][fname]
        # filter to time periods dict
        for sname,chunk in periods.items():
            # load sub bit of the stripe
            time,i0,i1 = loadTDMSSub(fn,chunk,is_time=True)
            #t=time
            #f,t,Zxx = plt.spectrogram(i0,1e6)
            Zxx_0, f, t, im = plt.specgram(i0, NFFT=8224, Fs=1e6) #8224
            fig1,ax = plt.subplots(constrained_layout=True)
            if shift:
                #plt.colorbar(ax.pcolormesh(t, np.fft.fftshift(f), np.fft.fftshift(Zxx, axes=0)**2),norm=colors.LogNorm())
                plt.colorbar(ax.pcolormesh(t, np.fft.fftshift(f), np.fft.fftshift(10*np.log10(Zxx_0), axes=0)), norm=colors.LogNorm())
            else:
                #plt.colorbar(ax.pcolormesh(t, f, Zxx ** 2),norm=colors.LogNorm())
                plt.colorbar(ax.pcolormesh(t, f, np.log10(Zxx_0) ),norm=colors.LogNorm())
            ax.set(xlabel="Time (s)",ylabel="Frequency (Hz)",title=f"{fname},{sname} Input 0")
           
            #f,t,Zxx = signal.spectrogram(i1,1e6)
            Zxx_1, f, t, im = plt.specgram(i1, NFFT=8224, Fs=1e6)
            fig2,ax = plt.subplots(constrained_layout=True)
            
            if shift:
                #plt.colorbar(ax.pcolormesh(t, np.fft.fftshift(f), np.fft.fftshift(Zxx, axes=0)**2),norm=colors.LogNorm())
                plt.colorbar(ax.pcolormesh(t, np.fft.fftshift(f), np.fft.fftshift(10*np.log10(Zxx_1), axes=0)),norm=colors.LogNorm())
            else:
                #plt.colorbar(ax.pcolormesh(t, f, Zxx ** 2),norm=colors.LogNorm())
                plt.colorbar(ax.pcolormesh(t, f, np.log10(Zxx_1) ),norm=colors.LogNorm())
            ax.set(xlabel="Time (s)",ylabel="Frequency (Hz)",title=f"{fname},{sname} Input 1")
            #fig.savefig(fr"{os.path.splitext(fn)[0]}-Input 1-spectrogram-{sname}{'-shifted' if shift else ''}.png")
            #plt.close(fig)
            Zxx_i0 = np.append(Zxx_i0,Zxx_0)
            Zxx_i1 = np.append(Zxx_i1,Zxx_1)
            #print(len(Zxx))
            t = np.append(t,t)
            f = np.append(f,f)
        #return fig1, fig2
        for x in Zxx_i0:
            Zxx_newi0 = np.append(Zxx_newi0,(Zxx_i0[1] - Zxx_i0[0]))
        print(len(Zxx_newi0))
        for x in Zxx_i1:
            Zxx_newi1 = np.append(Zxx_newi1,(Zxx_i1[1] - Zxx_i1[0]))
        fig3,ax = plt.subplots(constrained_layout=True)
        #plt.plot(t,Zxx_newi0)
        plt.colorbar(ax.pcolormesh(t[1], f[1], np.log10(Zxx_newi0) ),norm=colors.LogNorm())
        ax.set(xlabel="Time (s)",ylabel="Frequency (Hz)")
        fig4,ax = plt.subplots(constrained_layout=True)
        plt.colorbar(ax.pcolormesh(t[1], f[1], np.log10(Zxx_newi1) ),norm=colors.LogNorm())
        ax.set(xlabel="Time (s)",ylabel="Frequency (Hz)")
    return fig3, fig4

def bandpassfilter(fn,freq = [100e3, 250e3],mode=['highpass','lowpass'],**kwargs):  #(fn,freq=50e3,mode='highpass',**kwargs)
    '''
        Apply filters to each identified stripe in the file

        A filter can be applied to the signal using the mode, order and and cutoff_freq keywords.
        The cutoff_freq keyword is the cutoff frequency of the filter.
        Can either be:
            - Single value representing a highpass/lowpass filter
            - 2-element Tuple/list for a bandpass filter.
            - List of values or list/tuples for a series of filters applied sequentially

        The mode parameter is to specify whether the filters are lowpass ("lp") or highpass ("hp"). If it's a single string, then it's applied
        to all non-bandpass filters in cutoff_freq. If it's a list, then it MUST be the same length as the number of filters

        Generated plots are saved in the same location as the source file

        This is intended to remove the noise floor.

        Inputs:
            fn : TDMS path
            freq : Cutoff freq. Float, list of floats or list of tuples/lists. Default 50e3.
            order : Filter order. Default 10.
    '''
    fname = os.path.splitext(os.path.basename(fn))[0]
    # check if supported
    #periods = list(filter(lambda x : os.path.splitext(os.path.basename(path))[0] in x,PERIODS))
    #feedrate = list(filter(lambda x : os.path.splitext(os.path.basename(path))[0] in x,FEED_RATE))
    #if feedrate:
    #    feedrate = feedrate[0][os.path.splitext(os.path.basename(path))[0]] #Select Feedrate base on the source file
    #if periods:
     #   periods = periods[0][os.path.splitext(os.path.basename(path))[0]]
    periods = list(filter(lambda x : fname in x,PERIODS))
    #fs = 1e6
    #lowcut = 50e3
    #highcut = 300e3
    freq = [100e3, 250e3]
    if periods:
        periods = periods[0][fname]
        # ensure frequencies and modes are lists
        if isinstance(freq,(int,float)):
            freq = [freq,]
        # if mode is a single string them treat all filters as that mode
        if isinstance(mode,str):
            modes = len(freq)*[mode,]
        elif len(freq) != len(mode):
            raise ValueError("Number of modes must match the number of filters!")
        else:
            modes = list(mode)
        # format frequencies and modes into strings for plotting titles and saving
        modes_save_string = '-'.join(modes)
        modes_plot_string = ','.join(modes)
        freq_save_string = '-'.join([str(c) for c in freq])
        freq_plot_string = ','.join([str(c) for c in freq])
        # filter to time periods dict
        for sname,chunk in periods.items():
            # load sub bit of the stripe
            time,i0,i1 = loadTDMSSub(fn,chunk,is_time=True)
            # plot period
            f_i0,ax_i0 = plt.subplots(constrained_layout=True)
            ax_i0.plot(time,i0,'b-',label="Original")
            # plot period
            f_i1,ax_i1 = plt.subplots(constrained_layout=True)
            ax_i1.plot(time,i1,'b-',label="Original")
            # iterate over each filter and mode            
            #for c,m in zip(freq,modes):
            #   print(sname,c,m)
               # if it's a single value then it's a highpass/lowpass filter
            #   if isinstance(c,(int,list)):
            print("bing!")
            sos = butter(kwargs.get("order",10), 100e3, 'highpass', fs=1e6, output='sos',analog=False)
            i0 = sosfilt(sos, i0)
            i1 = sosfilt(sos, i1)
            sos = butter(kwargs.get("order",10), 250e3, 'lowpass', fs=1e6, output='sos',analog=False)
            i0 = sosfilt(sos, i0)
            i1 = sosfilt(sos, i1)
               # if it's a list/tuple then it's a bandpass filter
             #  elif isinstance(c,(tuple,list)):
               #    print("bong!")
              #     if m == "bandpass":
               #        i0 = butter_bandpass_filter(i0,c[0],c[1],kwargs.get("order",10))
               ##        i1 = butter_bandpass_filter(i1,c[0],c[1],kwargs.get("order",10))
               #    elif m == "bandstop":
               #        i0 = butter_bandstop_filter(i0,c[0],c[1],kwargs.get("order",10))
                #       i1 = butter_bandstop_filter(i1,c[0],c[1],kwargs.get("order",10))
           #print(max(i1, key=abs))
           #plot data
            ax_i0.plot(time,i0,'r-',label="Filtered")
            ax_i0.legend()
            ax_i0.set(xlabel="Time (s)",ylabel="Voltage (V)",title=f"{fname}, Input 0, {sname}, {modes_plot_string}, {freq_plot_string}Hz")
            #f_i0.savefig(fr"{os.path.splitext(fn)[0]}-Input 0-time-stripe-{sname}-filtered-{modes_save_string}-freq-{freq_save_string}.png")
            #plt.close(f_i0)
            #plot data
            
            ax_i1.plot(time,i1,'r-',label="Filtered")
            ax_i1.legend()
            ax_i1.set(xlabel="Time (s)",ylabel="Voltage (V)",title=f"{fname}, Input 1, {sname}, {modes_plot_string}, {freq_plot_string}Hz")
            #f_i1.savefig(fr"{os.path.splitext(fn)[0]}-Input 1-time-stripe-{sname}-filtered-{modes_save_string}-freq-{freq_save_string}.png")
            #plt.close(f_i1)
        i0 = i0.append(i0)
        i1 = i1.append(i1)
            #return f_i0, f_i1
    else:
        print(f"Unsupported file {fn}!")

def FilterSpecgram(fn,shift=False):
    '''
        Plot the spectrogram of each stripe in the target file

        Each file is 

        Inputs:
            fn : TDMS path
    '''
    from scipy import signal
    #from scipy.signal import spectrogram
    import matplotlib.pyplot as plt
    import matplotlib.colors as colors
    #from matplotlib.colors import LogNorm
    fname = os.path.splitext(os.path.basename(fn))[0]
    # check if supported
    Zxx_i0 = []
    Zxx_i1 = []
    Zxx_newi0 = []
    Zxx_newi1 = []
    t = []
    f = []
    
    time, i0, i1 = bandpassfilterStripes(fn, freq=[100e3, 250e3])
    
    #periods = list(filter(lambda x : fname in x,PERIODS))
    #if periods:
    #    periods = periods[0][fname]
    #   # filter to time periods dict
    for sname,chunk in periods.items():
            # load sub bit of the stripe
    #        time,i0,i1 = loadTDMSSub(fn,chunk,is_time=True)
            #t=time
            #f,t,Zxx = plt.spectrogram(i0,1e6)
            Zxx_0, f, t, im = plt.specgram(i0, NFFT=8224, Fs=1e6) #8224
            fig1,ax = plt.subplots(constrained_layout=True)
            if shift:
                #plt.colorbar(ax.pcolormesh(t, np.fft.fftshift(f), np.fft.fftshift(Zxx, axes=0)**2),norm=colors.LogNorm())
                plt.colorbar(ax.pcolormesh(t, np.fft.fftshift(f), np.fft.fftshift(10*np.log10(Zxx_0), axes=0)), norm=colors.LogNorm())
            else:
                #plt.colorbar(ax.pcolormesh(t, f, Zxx ** 2),norm=colors.LogNorm())
                plt.colorbar(ax.pcolormesh(t, f, np.log10(Zxx_0) ),norm=colors.LogNorm())
            ax.set(xlabel="Time (s)",ylabel="Frequency (Hz)",title=f"{fname},{sname} Input 0")
           
            #f,t,Zxx = signal.spectrogram(i1,1e6)
            Zxx_1, f, t, im = plt.specgram(i1, NFFT=8224, Fs=1e6)
            fig2,ax = plt.subplots(constrained_layout=True)
            
            if shift:
                #plt.colorbar(ax.pcolormesh(t, np.fft.fftshift(f), np.fft.fftshift(Zxx, axes=0)**2),norm=colors.LogNorm())
                plt.colorbar(ax.pcolormesh(t, np.fft.fftshift(f), np.fft.fftshift(10*np.log10(Zxx_1), axes=0)),norm=colors.LogNorm())
            else:
                #plt.colorbar(ax.pcolormesh(t, f, Zxx ** 2),norm=colors.LogNorm())
                plt.colorbar(ax.pcolormesh(t, f, np.log10(Zxx_1) ),norm=colors.LogNorm())
            ax.set(xlabel="Time (s)",ylabel="Frequency (Hz)",title=f"{fname},{sname} Input 1")
            #fig.savefig(fr"{os.path.splitext(fn)[0]}-Input 1-spectrogram-{sname}{'-shifted' if shift else ''}.png")
            #plt.close(fig)
           # Zxx_i0 = np.append(Zxx_i0,Zxx_0)
           # Zxx_i1 = np.append(Zxx_i1,Zxx_1)
            #print(len(Zxx))
           # t = np.append(t,t)
           # f = np.append(f,f)
    return fig1, fig2
       # for x in Zxx_i0:
       #     Zxx_newi0 = np.append(Zxx_newi0,(Zxx_i0[1] - Zxx_i0[0]))
        #print(len(Zxx_newi0))
        #for x in Zxx_i1:
        #    Zxx_newi1 = np.append(Zxx_newi1,(Zxx_i1[1] - Zxx_i1[0]))
        #fig3,ax = plt.subplots(constrained_layout=True)
        #plt.plot(t,Zxx_newi0)
        #plt.colorbar(ax.pcolormesh(t[1], f[1], np.log10(Zxx_newi0) ),norm=colors.LogNorm())
        #ax.set(xlabel="Time (s)",ylabel="Frequency (Hz)")
        #fig4,ax = plt.subplots(constrained_layout=True)
        #plt.colorbar(ax.pcolormesh(t[1], f[1], np.log10(Zxx_newi1) ),norm=colors.LogNorm())
        #ax.set(xlabel="Time (s)",ylabel="Frequency (Hz)")
    #return fig3, fig4

def plotSTFTPLT(signal,nperseg=2560,fclip=None,use_log=True,**kwargs):
    '''
        Plot the STFT of the target signal using Matplotlib

        fclip is for clipping the freq to above the set threshold
        as contrast over the full range can be an issue

        Inputs:
            signal : Numpy array to perform stft on
            nperseg : Number of segments. See scipy.signal.stft
            fclip : Freq clipping threshold. Default None
            use_log : Flag to use log yaxis. Default True,
            theme : Seaborn threshold to use

        Returns generated figure
    '''
    #fname = os.path.splitext(os.path.basename(signal))[0]
    import matplotlib.colors as colors
    # set seaborne theme
    sns.set_theme(kwargs.get("theme","paper"))
    # perform STFT at the set parameters
    f, t, Zxx = stft(signal, 1e6, nperseg=nperseg)
    # offset time vector for plotting
    t += kwargs.get("tmin",0.0)
    # if user wants the frequency clipped above a particular point
    if fclip:
        #Zxx = Zxx[f>=fclip,:]
        Zxx = Zxx[(f>=fclip[0]) & (f<=fclip[1])]
        f = f[(f>=fclip[0]) & (f<=fclip[1])]
    #np.savetxt(f"Stripe1_15gmin_Layer4_Input_0_Data_Zxx1.csv", 
    #Zxx, delimiter =", ",  # Set the delimiter as a comma followed by a space
    #fmt ='% s')
    ##np.savetxt(f"Stripe1_15gmin_Layer1_Input_0_Data_t.csv",   #Commented out 20240119
    ##t, delimiter =", ",  # Set the delimiter as a comma followed by a space
    ##fmt ='% s')
    #np.savetxt(f"Stripe1_15gmin_Layer4_Input_0_Data_f1.csv", 
    #f, delimiter =", ",  # Set the delimiter as a comma followed by a space
    #fmt ='% s')
    
        #Zxx = Zxx - 0.1*max(Zxx, key=abs) #Subtract 10% of max value of signal as soft-thresholding
    amp = 2 * np.sqrt(2)
    #Zxx = np.where(np.abs(Zxx) >= amp/10, Zxx, 0)
    _, xrec = istft(Zxx, 1e6)
    #print(len(xrec))
    max_value = np.max(np.abs(Zxx))
    min_value = np.min(np.abs(Zxx))
    Zxx = Zxx/max_value
    X_raw = [t, f, Zxx]
    
    
    
   # with h5py.File('f"{os.path.splitext(os.path.basename(fn))[0]}, Input 0, {sname} STFT".h5','w') as hf:
        # Create a group to store the spectrogram data
        # spectrogram_group = hf.create_group('spectrogram_data_15gmin_Layer2_Input0')
        #
        # # Save individual arrays within the group
        # hf.create_dataset('time', data=t)
        # spectrogram_group.create_dataset('frequency', data=f)
        # spectrogram_group.create_dataset('Zxx', data=Zxx)
    ## Added for exporting normalised data for machine learning attempts 20240119
#    with h5py.File('Coating2_Stripe1_15gmin_Layer2_Input0_Norm_Data2.h5','w') as hf:
#        # Create a group to store the spectrogram data
#        spectrogram_group = hf.create_group('spectrogram_data_15gmin_Layer2_Input0')
#
#        # Save individual arrays within the group
#        spectrogram_group.create_dataset('time', data=t)
#        spectrogram_group.create_dataset('frequency', data=f)
#        spectrogram_group.create_dataset('Zxx', data=Zxx)
#    np.savetxt(f"Stripe5_35gmin_Layer1_Input_0_Data_Zxx_2024.csv", 
#    Zxx, delimiter =", ",  # Set the delimiter as a comma followed by a space
#    fmt ='% s')
#    np.savetxt(f"Stripe5_35gmin_Layer1_Input_0_Data_t_2024.csv",   #Commented out 20240119
#    t, delimiter =", ",  # Set the delimiter as a comma followed by a space
#    fmt ='% s')
#    np.savetxt(f"Stripe4_30gmin_Layer1_Input_0_Data_f1_2024.csv", 
#    f, delimiter =", ",  # Set the delimiter as a comma followed by a space
#    fmt ='% s')
    
    ##
    fig,ax = plt.subplots(constrained_layout=True)
    #fig,ax = plt.subplots(subplot_kw={"projection": "3d"})
    # if using colormap log Norm
    if use_log:
        #X = ax.pcolormesh(t, f, np.abs(Zxx), norm=colors.LogNorm(vmin=np.abs(Zxx).max()*0.01, vmax=np.abs(Zxx).max()),cmap=kwargs.get("cmap",'inferno'),shading=kwargs.get("shading",'gouraud'))
        ##X = ax.pcolormesh(t, f, np.abs(Zxx), norm=colors.LogNorm(vmin=0.1e-4, vmax=10**(-2)),cmap=kwargs.get("cmap",'inferno'),shading=kwargs.get("shading",'gouraud'))
        X = ax.pcolormesh(t, f, np.abs(Zxx), norm=colors.LogNorm(vmin=1e-3, vmax=1),cmap=kwargs.get("cmap",'cool'),shading=kwargs.get("shading",'gouraud'))
        #X = ax.pcolormesh(f, t, np.abs(Zxx), cmap=kwargs.get("cmap",'inferno'),shading=kwargs.get("shading",'gouraud'))
        #X = ax.pcolormesh(t, f, np.abs(Zxx**2), norm=colors.LogNorm(vmin=np.abs(Zxx**2).max()*0.0005, vmax=np.abs(Zxx**2).max()),cmap=kwargs.get("cmap",'inferno'),shading=kwargs.get("shading",'gouraud'))
        #X = ax.pcolormesh(t, f, np.abs(Zxx), norm=colors.LogNorm(vmin=2e-4, vmax=3e-3),cmap=kwargs.get("cmap",'inferno'),shading=kwargs.get("shading",'gouraud'))
    else:
        X = ax.pcolormesh(t, f, np.abs(Zxx),cmap=kwargs.get("cmap",'inferno'),shading=kwargs.get("shading",'gouraud'))
    # set title
    ax.set_title(kwargs.get("title",'STFT Magnitude'))
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    # set colorbar
    plt.colorbar(X)
    print(len(t))
    print(len(f))
    print(len(Zxx))
    
    #fig1, ax1 = plt.subplots(constrained_layout=True)
    #ax1.plot(_, xrec) #signal, t,
    #ax1.set_xlabel('Time [sec]')
    #ax1.set_ylabel('Voltage [V]')
    #ax1.legend('Raw signal', 'Filtered signal')
    #return fig#, fig1
    #plt.show()

def plotSTFTStripes(path,nperseg=2560,fclip=None,use_log=True,**kwargs):
    '''
        Plot the STFT of each stripe in each channel

        This relies on the stripe time periods being listed in STRIPE_PERIOD and the file being supported

        fclip is the lower frequency to threshold above. This is to avoid the noise floor of signals and focus non the interesting stuff.

        When use_log is True, matplotlib.colors.LogNorm is being used

        Figures are saved to the same location at the source file

        Input:
            path : TDMS path
            nperseg : Number of points per segment
            fclip : Frequency to clip the STFT above
            use_log : Use log y-axis
            sname: Stripe name
    '''
    sns.set_theme(kwargs.get("theme","paper"))
    ftitle = os.path.basename(os.path.splitext(path)[0])
    labels = ["15 gmin", "20 gmin","25 gmin","30 gmin", "35 gmin"]#,"15 gmin", "20 gmin","25 gmin","30 gmin", "35 gmin","15 gmin", "20 gmin","25 gmin","30 gmin", "35 gmin","15 gmin", "20 gmin","25 gmin","30 gmin", "35 gmin",]
    # for each tdms
    for fn in glob(path):
        periods = list(filter(lambda x : os.path.splitext(os.path.basename(fn))[0] in x,PERIODS))
        if periods:
            periods = periods[0][os.path.splitext(os.path.basename(fn))[0]]
            # filter to time periods dict
            # Create an HDF5 file for storing STFT results
            hdf5_filename = 'stft_results_20240206_C1Test.h5'
            with h5py.File(hdf5_filename, 'w') as hdf5_file:
            
                for i, (sname,chunk) in enumerate(periods.items()):
                    # load sub bit of the stripe
                    time,i0,i1 = loadTDMSSub(path,chunk,is_time=True)
                
                    #f=plotSTFTPLT(i0,nperseg,fclip,use_log,title=f"{os.path.splitext(os.path.basename(fn))[0]}, Input 0, {sname} STFT",tmin=chunk[0],**kwargs)
                                        
                   ##fig,ax = plt.subplots(constrained_layout=True)
                   # perform STFT at the set parameters
                    f, t, Zxx = stft(i0, 1e6, nperseg=nperseg)
                   # offset time vector for plotting
                   #t += kwargs.get("tmin",0.0)
                   # if user wants the frequency clipped above a particular point
                    if fclip:
                       #Zxx = Zxx[f>=fclip,:]
                       Zxx = Zxx[(f>=fclip[0]) & (f<=fclip[1])]
                       f = f[(f>=fclip[0]) & (f<=fclip[1])]
                    max_value = np.max(np.abs(Zxx))
                    min_value = np.min(np.abs(Zxx))
                    Zxx = Zxx/max_value
                    
                    # Create a group for each data entry
                    group = hdf5_file.create_group(sname)
                    
                    # Save STFT results and label to the group
                    group.create_dataset('time', data=t)
                    group.create_dataset('frequency', data=f)
                    group.create_dataset('intensity', data=np.abs(Zxx))  # Using absolute value for intensity
                    
                    
                    # Assign label based on the index
                    if i < len(labels):
                        label = labels[i]
                    else:
                        label = 'Unknown'
                    # Add label as an attribute to the group
                    group.attrs['label'] = sname
                    #print(label)
                    #X_raw = [t, f, Zxx]
                    ##fig,ax = plt.subplots(constrained_layout=True)
            print(f"STFT results saved to {hdf5_filename}")     
#                   if use_log:
#                       ax.pcolormesh(t, f, np.abs(Zxx), norm=colors.LogNorm(vmin=1e-3, vmax=1),cmap=kwargs.get("cmap",'cool'),shading=kwargs.get("shading",'gouraud'))
#                   else:
#                       ax.pcolormesh(t, f, np.abs(Zxx),cmap=kwargs.get("cmap",'inferno'),shading=kwargs.get("shading",'gouraud'))
#                   # set title
#                   ax.set(xlabel = 'Time [sec]', ylabel = 'Frequency [Hz]', title = f"{ftitle}\nInput 0 {sname} STFT")
#                
#                   ## Added for exporting normalised data for machine learning attempts 20240119
#                   with h5py.File('Coating2_Input0_Norm_Data2.h5','w') as hf:
#                       # Create a group to store the spectrogram data
#                       spectrogram_group = hf.create_group(f"{ftitle}\nInput 0 {sname} STFT")
#               
#                       # Save individual arrays within the group
#                       spectrogram_group.create_dataset('time', data=t)
#                       spectrogram_group.create_dataset('frequency', data=f)
#                       spectrogram_group.create_dataset('Zxx', data=Zxx)
#                
                    #ax.set_ylabel()
                    #ax.set_xlabel()
                    # perform stft for each channel
                    #f1=plotSTFTPLT(i0,nperseg,fclip,use_log,title=f"{os.path.splitext(os.path.basename(fn))[0]}, Input 0, {sname} STFT",tmin=chunk[0],**kwargs)
                                
                    #if fclip:
                    #    f1.savefig(fr"{os.path.splitext(fn)[0]}-Input 0-stft-stripe-{sname}-clip-{fclip}.png")
                    #else:
                    #    f1.savefig(fr"{os.path.splitext(fn)[0]}-Input 0-stft-stripe-{sname}.png")
                    #plt.close(f1)

                    # perform stft for each channel
                    #f2=plotSTFTPLT(i1,nperseg,fclip,use_log,title=f"{os.path.splitext(os.path.basename(fn))[0]}, Input 1, {sname} STFT",tmin=chunk[0],**kwargs)
                
                    #if fclip:
                    #    f2.savefig(fr"{os.path.splitext(fn)[0]}-Input 1-stft-stripe-{sname}-clip-{fclip}.png")
                    #else:
                    #    f2.savefig(fr"{os.path.splitext(fn)[0]}-Input 1-stft-stripe-{sname}.png")
                    #plt.close(f2)
                    #return fig#, f2
        else:
            print(f"Unsupported file {fn}!")



def plotSTFT(path,nperseg=2560,fclip=None,use_log=True,**kwargs):
    '''
        Plot the STFT of each stripe in each channel

        This relies on the stripe time periods being listed in STRIPE_PERIOD and the file being supported

        fclip is the lower frequency to threshold above. This is to avoid the noise floor of signals and focus non the interesting stuff.

        When use_log is True, matplotlib.colors.LogNorm is being used

        Figures are saved to the same location at the source file

        Input:
            path : TDMS path
            nperseg : Number of points per segment
            fclip : Frequency to clip the STFT above
            use_log : Use log y-axis
    '''
    import matplotlib.colors as colors
    #from scipy import signal
    # set seaborne theme
    sns.set_theme(kwargs.get("theme","paper"))
    # for each tdms
    for fn in glob(path):
        periods = list(filter(lambda x : os.path.splitext(os.path.basename(fn))[0] in x,PERIODS))
        if periods:
            periods = periods[0][os.path.splitext(os.path.basename(fn))[0]]
            # filter to time periods dict
            for sname,chunk in periods.items():
                time,i0,i1 = loadTDMSSub(path,chunk,is_time=True)
                # perform stft for each channel
                
                
                # perform STFT at the set parameters
                fi0, ti0, Zxx_i0 = stft(i0, 1e6, nperseg=nperseg)
                # offset time vector for plotting
                ti0 += kwargs.get("tmin",0.0)
                # if user wants the frequency clipped above a particular point
                if fclip:
                    Zxx_i0 = Zxx_i0[fi0>=fclip,:]
                    fi0 = fi0[fi0>=fclip]
                #Zxx = Zxx - 0.1*max(Zxx, key=abs) #Subtract 10% of max value of signal as soft-thresholding
                amp = 2 * np.sqrt(2)
                #Zxx1 = np.where(np.abs(Zxx) >= amp/10, Zxx, 0)
                _, xrec_i0 = istft(Zxx_i0, 1e6)
                
                                
                # perform STFT at the set parameters
                fi1, ti1, Zxx_i1 = stft(i1, 1e6, nperseg=nperseg)
                # offset time vector for plotting
                ti1 += kwargs.get("tmin",0.0)
                # if user wants the frequency clipped above a particular point
                if fclip:
                    Zxx_i1 = Zxx_i1[fi1>=fclip,:]
                    fi1 = fi1[fi1>=fclip]
                #Zxx = Zxx - 0.1*max(Zxx, key=abs) #Subtract 10% of max value of signal as soft-thresholding
                amp = 2 * np.sqrt(2)
                #Zxx1 = np.where(np.abs(Zxx) >= amp/10, Zxx, 0)
                _, xrec_i1 = istft(Zxx_i1, 1e6)
                
                np.savetxt("Zxx_Coating2,Stripe_1_1 Input 0.csv", 
                Zxx_i0, delimiter =", ",  # Set the delimiter as a comma followed by a space
                fmt ='% s')
                np.savetxt("f_Coating2,Stripe_1_1 Input 0.csv", 
                fi0, delimiter =", ",  # Set the delimiter as a comma followed by a space
                fmt ='% s')
                
                np.savetxt("Zxx_Coating2,Stripe_1_1 Input 1.csv", 
                Zxx_i1, delimiter =", ",  # Set the delimiter as a comma followed by a space
                fmt ='% s')
                np.savetxt("f_Coating2,Stripe_1_1 Input 1.csv", 
                fi1, delimiter =", ",  # Set the delimiter as a comma followed by a space
                fmt ='% s')
                
                #f1=plotSTFTPLT(i0,nperseg,fclip,use_log,title=f"{os.path.splitext(os.path.basename(fn))[0]}, Input 0, {sname} STFT",tmin=chunk[0],**kwargs)
                #np.savetxt("Zxx_Coating2,Ref Input 0.csv", 
                #i0, delimiter =", ",  # Set the delimiter as a comma followed by a space
                #fmt ='% s')
                
                #if fclip:
                #    f1.savefig(fr"{os.path.splitext(fn)[0]}-Input 0-stft-stripe-{sname}-clip-{fclip}.png")
                #else:
                #    f1.savefig(fr"{os.path.splitext(fn)[0]}-Input 0-stft-stripe-{sname}.png")
                #plt.close(f1)

                # perform stft for each channel
                #f2=plotSTFTPLT(i1,nperseg,fclip,use_log,title=f"{os.path.splitext(os.path.basename(fn))[0]}, Input 1, {sname} STFT",tmin=chunk[0],**kwargs)
                #np.savetxt("Zxx_Coating2,Ref Input 1.csv", 
                #i1, delimiter =", ",  # Set the delimiter as a comma followed by a space
                #fmt ='% s')
                #if fclip:
                #    f2.savefig(fr"{os.path.splitext(fn)[0]}-Input 1-stft-stripe-{sname}-clip-{fclip}.png")
                #else:
                #    f2.savefig(fr"{os.path.splitext(fn)[0]}-Input 1-stft-stripe-{sname}.png")
                #plt.close(f2)
                #return f1, f2
        else:
            print(f"Unsupported file {fn}!")




if __name__ == "__main__":
    import matplotlib.colors as colors
    from scipy.signal import periodogram
    plt.rcParams['agg.path.chunksize'] = 10000

    #f = calcStripeAreas("ae/sheff_lsbu_stripe_coating_2.tdms")
    #f = filterStripesCalcArea("ae/sheff_lsbu_stripe_coating_2.tdms",freq=50e3,mode='highpass', dist=int(50e3))
    #bandpassfilterStripes("ae/sheff_lsbu_stripe_coating_2.tdms", freq=[100e3, 250e3])
    #plotSTFTStripes("ae/sheff_lsbu_stripe_coating_2.tdms", fclip=100e3, use_log=True)
    #plotSTFTStripes("ae/sheff_lsbu_stripe_coating_1.tdms", fclip=100e3, use_log=True)
    #plotSTFTStripes("ae/sheff_lsbu_stripe_coating_3_pulsing.tdms", fclip=[280e3,370e3], use_log=True)
   # plotSTFTStripes("ae/sheff_lsbu_stripe_coating_2.tdms", fclip=[280e3,370e3], use_log=True)
    plotSTFTStripes("ae/sheff_lsbu_stripe_coating_1.tdms", fclip=[100e3,500e3] , use_log=True) #[100e3,500e3]
    #plotSTFT("ae/sheff_lsbu_stripe_coating_2.tdms", fclip=100e3, use_log=True)
    #bandpassfilter("ae/sheff_lsbu_stripe_coating_2.tdms", freq=[100e3, 250e3])
    #stripeSpecgram("ae/sheff_lsbu_stripe_coating_2.tdms")
