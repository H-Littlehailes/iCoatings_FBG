# -*- coding: utf-8 -*-
"""
Created on Wed May  3 09:58:06 2023

@author: David Miller
"""

from nptdms import TdmsFile
import os
import seaborn as sns
import tempfile
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from scipy.signal import stft,butter,sosfilt,sosfreqz
#path = 'C:/Users/hugh/LSBU_Stripe_Data'
# time periods where the stripes occur in each file
CLIP_PERIOD = {'sheff_lsbu_stripe_coating_1' : [120.0,180.0],
               'sheff_lsbu_stripe_coating_2' : [100.0,350.0],
               'sheff_lsbu_stripe_coating_3_pulsing' : [100.0,180.0]}



# stripe locations 0-1000
STRIPE_PERIOD = {'sheff_lsbu_stripe_coating_1': {'stripe_1':[120, 125],
                                                'stripe_2':[127, 133],
                                                'stripe_3':[141, 146],
                                                'stripe_4':[151, 156],
                                                'stripe_5':[162.5, 167.5],
                                                'stripe_6':[175, 180]}}
                                                 # first pass
STRIPE_PERIOD_2 = {'sheff_lsbu_stripe_coating_2': {'stripe_1_1':[107, 112],
                                                'stripe_2_1':[118, 123],
                                                'stripe_3_1':[127, 133],
                                                'stripe_4_1':[137, 143],
                                                'stripe_5_1':[146, 152],
                                                # 2nd pass
                                                'stripe_1_2':[157, 163], 
                                                'stripe_2_2':[169, 175],
                                                'stripe_3_2':[180, 185.5],
                                                'stripe_4_2':[190, 195],
                                                'stripe_5_2':[199, 205],
                                                # 3rd pass
                                                'stripe_1_3':[215, 221],
                                                'stripe_2_3':[226, 231], 
                                                'stripe_3_3':[237, 243],
                                                'stripe_4_3':[249, 255],
                                                'stripe_5_3':[260, 266],
                                                # unknown
                                                'stripe_1_4':[277, 282],
                                                'stripe_2_4':[290, 296],
                                                'stripe_3_4':[304, 310],
                                                'stripe_4_4':[314, 320],
                                                'stripe_5_4':[327,333],
                                                 }}

STRIPE_PERIOD_3 = {'sheff_lsbu_stripe_coating_3_pulsing':{'stripe_1':[105, 110],
                                                'stripe_2':[112, 117],
                                                'stripe_3':[119, 124],
                                                'stripe_4':[125, 131],
                                                'stripe_5':[134, 139],
                                                'stripe_6':[141, 147],
                                                'stripe_7':[148, 154],
                                                'stripe_8':[157, 163],
                                                'stripe_9':[164, 170],
                                                'stripe_10':[172, 178],
                                                }}
PERIODS = [STRIPE_PERIOD,STRIPE_PERIOD_2,STRIPE_PERIOD_3]



## dict for mapping stripe number to feed rate
FEED_RATE_1 = {'sheff_lsbu_stripe_coating_1':   {'stripe_1':'15 G/MIN',
                                                'stripe_2':'15 G/MIN',
                                                'stripe_3':'20 G/MIN',
                                                'stripe_4':'25 G/MIN',
                                                'stripe_5':'30 G/MIN',
                                                'stripe_6':'35 G/MIN'}}

FEED_RATE_2 = {'sheff_lsbu_stripe_coating_2': {'stripe_1_1':'15 G/MIN',
                                                'stripe_2_1':'20 G/MIN',
                                                'stripe_3_1':'25 G/MIN',
                                                'stripe_4_1':'30 G/MIN',
                                                'stripe_5_1':'35 G/MIN',
                                                # 2nd pass
                                                'stripe_1_2':'15 G/MIN (2)',
                                                'stripe_2_2':'20 G/MIN (2)',
                                                'stripe_3_2':'25 G/MIN (2)',
                                                'stripe_4_2':'30 G/MIN (2)',
                                                'stripe_5_2':'35 G/MIN (2)',
                                                # 3rd pass
                                                'stripe_1_3':'15 G/MIN (3)',
                                                'stripe_2_3':'20 G/MIN (3)',
                                                'stripe_3_3':'25 G/MIN (3)',
                                                'stripe_4_3':'30 G/MIN (3)',
                                                'stripe_5_3':'35 G/MIN (3)',
                                                # unknown
                                                'stripe_1_4':'15 G/MIN (4)',
                                                'stripe_2_4':'20 G/MIN (4)',
                                                'stripe_3_4':'25 G/MIN (4)',
                                                'stripe_4_4':'30 G/MIN (4)',
                                                'stripe_5_4':'35 G/MIN (4)',
                                                 }}

FEED_RATE = [FEED_RATE_1, FEED_RATE_2]


## from https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
# band pass
def butter_bandpass(lowcut, highcut, order=5):
    nyq = 0.5 * 1e6
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='bandpass', analog=False, output='sos')

def butter_bandpass_filter(data, lowcut, highcut, order=5):
    sos = butter_bandpass(lowcut, highcut, 1e6, order=order)
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

def applyFilters(data,freq,mode):
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
    if isinstance(freq,(int,float)):
        freq = [freq,]
    # if mode is a single string them treat all filters as that mode
    if isinstance(mode,str):
        modes = len(freq)*[mode,]
    elif len(freq) != len(mode):
        raise ValueError("Number of modes must match the number of filters!")
    else:
        modes = list(mode)
    # iterate over each filter and mode
    for c,m in zip(freq,modes):
        # if it's a single value then it's a highpass/lowpass filter
        if isinstance(c,(int,float)):
            sos = butter(kwargs.get("order",10), c/1e6, m, fs=1e6, output='sos',analog=False)
            data = sosfilt(sos, data)
        # if it's a list/tuple then it's a bandpass filter
        elif isinstance(c,(tuple,list)):
            if m == "bandpass":
                data = butter_bandpass_filter(data,c[0],c[1],kwargs.get("order",10))
            elif m == "bandstop":
                data = butter_bandpass_filter(data,c[0],c[1],kwargs.get("order",10))
    return data

def plotFreqResponse(freq,btype,pts=int(1e6/4),**kwargs):
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
    if isinstance(freq,(float,int)):
        freq_norm = freq/(1e6/2)
    else:
        freq_norm = freq
    # if low or highpass
    if btype in ["lowpass","highpass"]:
        sos = butter(kwargs.get("order",10),freq_norm,btype,analog=False,output='sos')
    # for band filters
    elif btype == "bandpass":
        sos = butter_bandpass(freq[0],freq[1],1e6,kwargs.get("order",10))
    elif btype == "bandstop":
        sos = butter_bandstop(freq[0],freq[1],1e6,kwargs.get("order",10))
    # generate freq response
    w,h = sosfreqz(sos,worN=pts,fs=1e6)
    f,ax = plt.subplots(nrows=2,constrained_layout=True)
    ax[0].plot(w,np.abs(h))
    ax[1].plot(w,np.angle(h))
    ax[0].set(xlabel="Frequency (Hz)",ylabel="Gain")
    ax[1].set(xlabel="Frequency (Hz)",ylabel="Phase (radians)")
    ax[0].vlines(freq,0,1,color='purple')
    ax[1].vlines(freq,-2*np.pi,2*np.pi,color='purple')
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
    data.rename(columns={c:c.split('/')[-1].strip("'") for c in data.columns},inplace=True)
    return data

def loadTDMSSub(path,chunk,is_time=False):
    '''
        Load in a sub-section of TDMS data
        Helps avoid loading in the entire file
        
    '''
    with TdmsFile(path) as tdms_file:
        # find group that contains recording
        group = list(filter(lambda x : "Recording" in x.name,tdms_file.groups()))[0]
        nf=group['Input 0'].data.shape[0]
        mt = nf*1e-6
        # convert to index
        if is_time:
            chunk = [max(0,min(nf,int(c*1e6))) for c in chunk]
        nf = group['Input 0'][chunk[0]:chunk[1]].shape[0]
        time = np.arange(nf)*1e-6
        time += chunk[0]*1e-6
        return time,group['Input 0'][chunk[0]:chunk[1]],group['Input 1'][chunk[0]:chunk[1]]
        
def replotAE(path,clip=True,ds=100):
    '''
        Replot the TDMS Acoutic Emission files
        If the file is supported in CLIP_PERIOD dictionary at the top of the file, then  it is cropped to the target period else plotted at full res
        Inputs:
            path : Input file path to TDMS file
            clip : Flag to clip the data. If True, then CLIP_PERIOD is referenced. If a tuple, then it's taken as the time period to clip to
        Returns figures for Input 0 and Input 1 respectively
    '''
    sns.set_theme("talk")
    data = loadTDMSData(path)
    # clip to known activity
    if clip:
        if any([os.path.splitext(os.path.basename(path))[0] in k for k in CLIP_PERIOD.keys()]):
            data = data[(data['Time (s)'] >= CLIP_PERIOD[os.path.splitext(os.path.basename(path))[0]][0]) & (data['Time (s)'] <= CLIP_PERIOD[os.path.splitext(os.path.basename(path))[0]][1])]
        elif isinstance(clip,(tuple,list)):
            data = data[(data['Time (s)'] >= clip[0]) & (data['Time (s)'] <= clip[1])]
        else:
            print(f"Failed to clip file {path}!")

    if isinstance(ds,(int,float)) and (not (ds is None)):
        data = data.iloc[::ds]
    # plot both channels
    f0,ax0 = plt.subplots(constrained_layout=True,figsize=(9,8))
    sns.lineplot(data=data,x='Time (s)',y='Input 0',ax=ax0)
    ax0.set_title(f"{os.path.splitext(os.path.basename(path))[0]}, Input 0")
    
    f1,ax1 = plt.subplots(constrained_layout=True,figsize=(9,8))
    sns.lineplot(data=data,x='Time (s)',y='Input 1',ax=ax1)
    ax1.set_title(f"{os.path.splitext(os.path.basename(path))[0]}, Input 1")
    return f0,f1
   
    
def plotRaster(path,nbins=1000,**kwargs):
    '''
        Plot the large dataset by rasterising into colours
        The parameter nbins controls the number of colours used
        Creates an image that's faster to produce than plotting the time series.
        The plot is saved to the same location as the source data
        Inputs:
            path : TDMS file path
            nbins : Number of colour bins
            cmap : Matplotlib colour map to use to convert data to colours
            bk : Background colour of plot
    '''
    import datashader as ds
    import colorcet as cc
    from fast_histogram import histogram2d
    import matplotlib.colors as colors
    data = loadTDMSData(path)
    for i,c in enumerate(data.columns):
        if c == "Time (s)":
            continue
        # from https://towardsdatascience.com/how-to-create-fast-and-accurate-scatter-plots-with-lots-of-data-in-python-a1d3f578e551
        cvs = ds.Canvas(plot_width=1000, plot_height=500)  # auto range or provide the `bounds` argument
        agg = cvs.points(data, 'Time (s)', 'Input 0')  # this is the histogram
        img = ds.tf.set_background(ds.tf.shade(agg, how="log", cmap=cc.fire), kwargs.get("bk","black")).to_pil()  # create a rasterized imageplt.imshow(img)
        # stack first column and time column
        X = np.hstack((data.values[:,i].reshape(-1,1),data.values[:,2].reshape(-1,1)))
        cmap = cc.cm[kwargs.get("cmap","fire")].copy()
        cmap.set_bad(cmap.get_under())  # set the color for 0 to avoid log(0)=inf
        # get bounds for axis
        bounds = [[X[:, 0].min(), X[:, 0].max()], [X[:, 1].min(), X[:, 1].max()]]
        # calculate 2d histogram for colour levels and let matplotlib handle the shading
        h = histogram2d(X[:, 0], X[:, 1], range=bounds, bins=nbins)
        f,ax = plt.subplots(constrained_layout=True)
        X = ax.imshow(h, norm=colors.LogNorm(vmin=1, vmax=h.max()), cmap=cmap)
        ax.axis('off')
        plt.colorbar(X)
        f.savefig(f"{os.path.splitext(path)[0]}-{c}.png")
        plt.close(f)