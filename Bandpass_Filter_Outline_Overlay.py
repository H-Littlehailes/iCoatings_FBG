from nptdms import TdmsFile
import os
import seaborn as sns
import tempfile
import numpy as np
from glob import glob
import matplotlib.pyplot as plt
from scipy.signal import stft,butter,sosfilt,sosfreqz

# time periods where the stripes occur in each file
CLIP_PERIOD = {'sheff_lsbu_stripe_coating_1' : [120.0,180.0],#180
               'sheff_lsbu_stripe_coating_2' : [80.0,350.0],
               'sheff_lsbu_stripe_coating_3_pulsing' : [100.0,180.0]}

# stripe locations 0-1000
STRIPE_PERIOD = {'sheff_lsbu_stripe_coating_1': {'stripe_1':[120, 125],
                                                'stripe_2':[127, 133],
                                                'stripe_3':[141, 146],
                                                'stripe_4':[151, 156],
                                                'stripe_5':[162.5, 167.5],
                                                'stripe_6':[175, 180]}}
                                                 # first pass
STRIPE_PERIOD_2 = {'sheff_lsbu_stripe_coating_2': {#'ref': [85, 90],                            Feed-rate comparison               Stripe comparison
                                                 #'stripe_1_1':[106.7, 111.7],  #                    100-250kHz: [107, 112]        250-350kHz: [106.7, 111.7]
                                                 #'stripe_2_1':[117.702, 122.702], #[118, 123],    100-250kHz: [117.85, 122.85]    250-350kHz: [117.702, 122.702]
                                                 #'stripe_3_1':[127.586, 132.586], #[127, 133],      100-250kHz: [127.7, 132.7]    250-350kHz: [127.586, 132.586]
                                                 #'stripe_4_1':[137.1, 142.1], #[137, 143],    100-250kHz: [137.22, 142.22]        250-350kHz: [137.1, 142.1]
                                                 'stripe_5_1':[146.82, 151.82], #[146, 152],    100-250kHz: [199.63, 204.63]       250-350kHz: [146.82, 151.82]
                                                # 2nd pass
                                                #'stripe_1_2':[157.2, 162.2],  #                  100-250kHz: [157.2, 162.2]       250-350kHz: [157.2, 162.2]
                                                #'stripe_2_2':[169.26, 174.26], #[169, 175],     100-250kHz: [169.26, 174.26]      250-350kHz: [169.26, 174.26]
                                                #'stripe_3_2':[180.18, 185.18], #[180, 185.5],   100-250kHz: [180.18, 185.18]      250-350kHz: [180.18, 185.18]
                                                #'stripe_4_2':[189.95, 194.95], #[190, 195],     100-250kHz: [189.85, 194.85]      250-350kHz: [189.95, 194.95]
                                                #'stripe_5_2':[199.854, 204.854], #[199.5, 205.5], 100-250kHz: [199.63, 204.63]     250-350kHz: [199.754, 204.754]
                                                # 3rd pass
                                                #'stripe_1_3':[214.83, 219.83], #[215, 221],   100-250kHz:[214.83, 219.83],        250-350kHz: [214.83, 219.83]
                                                #'stripe_2_3':[226.25, 231.25], #[226, 231],   100-250kHz:[226.25, 231.25]         250-350kHz: [226.25, 231.25]
                                                #'stripe_3_3':[237.259, 242.259], #[237, 243],   100-250kHz:[237.18, 242.18]       250-350kHz: [237.259, 242.259]
                                                #'stripe_4_3':[249.5, 254.5],#[249, 255],      100-250kHz:[249.5, 254.5]           250-350kHz: [249.5, 254.5]
                                                #'stripe_5_3':[260.595, 265.595],            #    100-250kHz:[260.41, 265.41]       250-350kHz: [260.745, 265.745]
                                                 #unknown
                                                #'stripe_1_4':[276.7, 281.7], #[277, 282],     100-250kHz: [276.7, 281.7]          250-350kHz: [276.7, 281.7]
                                                #'stripe_2_4':[290.641, 295.641], #[290, 296], 100-250kHz: [290.528, 295.528]      250-350kHz: [290.641, 295.641]
                                                #'stripe_3_4':[304.471, 309.471],          #    100-250kHz: [304.292, 309.292],    250-350kHz: [304.471, 309.471]
                                                #'stripe_4_4':[314.8, 319.8], #[314, 320],     100-250kHz: [314.8, 319.8]          250-350kHz: [314.8, 319.8]
                                                #'stripe_5_4':[327.563,332.563], #[327,333]    100-250kHz: [327.378,332.378]        250-350kHz: [327.378,332.378]
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

FEED_RATE_2 = {'sheff_lsbu_stripe_coating_2': {#'ref'       : 'ref',
                                                #'stripe_1_1':'15 G/MIN',
                                                #'stripe_2_1':'20 G/MIN',
                                                #'stripe_3_1':'25 G/MIN',
                                                #'stripe_4_1':'30 G/MIN',
                                                'stripe_5_1':'35 G/MIN',
                                                # 2nd pass
                                                #'stripe_1_2':'15 G/MIN (2)',
                                                #'stripe_2_2':'20 G/MIN (2)',
                                                #'stripe_3_2':'25 G/MIN (2)',
                                                #'stripe_4_2':'30 G/MIN (2)',
                                                #'stripe_5_2':'35 G/MIN (2)',
                                                # 3rd pass
                                                #'stripe_1_3':'15 G/MIN (3)',
                                                #'stripe_2_3':'20 G/MIN (3)',
                                                #'stripe_3_3':'25 G/MIN (3)',
                                                #'stripe_4_3':'30 G/MIN (3)',
                                                #'stripe_5_3':'35 G/MIN (3)',
                                                # unknown
                                                #'stripe_1_4':'15 G/MIN (4)',
                                                #'stripe_2_4':'20 G/MIN (4)',
                                                #'stripe_3_4':'25 G/MIN (4)',
                                                #'stripe_4_4':'30 G/MIN (4)',
                                                #'stripe_5_4':'35 G/MIN (4)',
                                                 }}

FEED_RATE = [FEED_RATE_1, FEED_RATE_2]


## from https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
# band pass
def butter_bandpass(lowcut, highcut, order=6):
    nyq = 0.5 * 1e6
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='bandpass', analog=False, output='sos')

def butter_bandpass_filter(data, lowcut, highcut, order=6):
    sos = butter_bandpass(lowcut, highcut, 1e6)
    return sosfilt(sos, data)

# band stop
def butter_bandstop(lowcut, highcut, order=5):
    nyq = 0.5 * 1e6
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='bandstop', analog=False, output='sos')

def butter_bandstop_filter(data, lowcut, highcut, order=5):
    sos = butter_bandstop(lowcut, highcut, 1e6)
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

def bandpassfilterOutline(path, freq=[50e3, 499e3], mode=['highpass', 'lowpass'], dist=int(10e3),model="separate", **kwargs):
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
    from scipy.signal import find_peaks
    import matplotlib.patches as mpatches
    from matplotlib.pyplot import cm
    fname = os.path.splitext(os.path.basename(path))[0]
    ftitle = os.path.splitext(os.path.basename(path))[0]
    add_label = ""
    add_label=f" Time={kwargs.get('time_period',None)}"
  #  check if supported
    for fn in glob(path):  
        periods = list(filter(lambda x : os.path.splitext(os.path.basename(path))[0] in x,PERIODS))
        feedrate = list(filter(lambda x : os.path.splitext(os.path.basename(path))[0] in x,FEED_RATE))
        if feedrate:
            feedrate = feedrate[0][ftitle] #Select Feedrate base on the source file
        if periods:
           periods = periods[0][ftitle]
       #periods = list(filter(lambda x: fname in x, PERIODS))
    #print(periods)
    # ref = list(filter(lambda y: 'ref' in y, PERIODS))

           freq = [50e3, 499e3]
  

              # periods2 = periods[0][fname1]
            #periods = periods[0][fname]
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
           n_time = [] # Empty list for time
           n_i0 = []   # Empty list for input 0 values
           n_i1 = []   # Empty list for input 1 values
           #ai0 = []    # Empty list for noise subtracted input 0 values
           #ai1 = []    # Empty list for noise subtracted input 1 values
           #ri0 = []
           #ri1 = []     
           max_positive_i0 = []
           max_negative_i0 = []
           max_positive_i1 = []
           max_negative_i1 = []
           
            # Loop through all listed stripe chunks and output the respective values to the
            # respective empty lists.
           f,ax = plt.subplots(ncols=2,constrained_layout=True,figsize=(14,6))
            #for x in periods:
           colours = ['r','y','g','c','b','m']
           ncolour = 0 
           for sname, chunk in periods.items(): # stripe name and chunk
           # load sub bit of the stripe
               time, i0, i1 = loadTDMSSub(path, chunk, is_time=True)
               #print(len(time))
              # n_time.append(time)
              # n_i0.append(i0)
              # n_i1.append(i1)
            #ai0 = n_i0[1] - n_i0[0]
            #ai1 = n_i1[1] - n_i1[0]
            
               sos = butter(kwargs.get("order", 10), 50e3, 'highpass',
                        fs=1e6, output='sos', analog=False)
               i0 = sosfilt(sos, i0)
               i1 = sosfilt(sos, i1)
               sos = butter(kwargs.get("order", 10), 499e3, 'lowpass',
                        fs=1e6, output='sos', analog=False)
               i0 = sosfilt(sos, i0)
               i1 = sosfilt(sos, i1)
            
            #sos = butter(kwargs.get("order", 10), 100e3, 'highpass',
            #             fs=1e6, output='sos', analog=False)
            #i0 = sosfilt(sos, ai0)
            #i1 = sosfilt(sos, ai1)
            #sos = butter(kwargs.get("order", 10), 250e3, 'lowpass',
            #             fs=1e6, output='sos', analog=False)
            #i0 = sosfilt(sos, i0)
            #i1 = sosfilt(sos, i1)
               
               #n = len(periods.items())
               #print(len(periods.items()))
               #color = cm.rainbow(np.linspace(0, 1, n))
               #colours = ['r','y','g','c','b','m']
               #ncolour = 0
               #for i, c in zip(range(n), color):
               #for i in (range(n)):
                   #color = colours[i]
               if model == "separate":
                       ## find edge of Input 0
                       # mask signal to +ve
                       mask_filt = i0.copy()
                       mask_filt[i0<0]=0 # positive
                       # find peaks in the signal
                       pks = find_peaks(mask_filt,distance=dist)[0]
                       if len(pks)==0:
                           raise ValueError(f"Failed to find +ve edge in Input 0!")
                       # plot the signal and +ve edge
                       #f,ax = plt.subplots(ncols=2,constrained_layout=True,figsize=(14,6))
                       #ax[0].plot(time,i0,'b-')
                       t = []
                       t = np.linspace(0, 5,len(i0[pks]))  
                       ax[0].plot(t,i0[pks],color=colours[ncolour],label=feedrate[sname])
                       max_positive_i0 = max(i0[pks])
                       #print("max_positive_i0")
                       #print(max_positive_i0)
                       
                       max_positive_i0_sum = sum(i0[pks]**2)
                     #  print("max_positive_i0_sum")
                     #  print(max_positive_i0_sum)
                       np.savetxt(f"Stripe5_35gmin_Input_0_Data_i0_pks_50_499kHz_Positive_Dist5e3.csv", 
                       i0[pks], delimiter =", ",  # Set the delimiter as a comma followed by a space
                       fmt ='% s')
                       np.savetxt(f"Stripe5_35gmin_Input_0_Data_t_50_499kHz_Positive_Dist5e3.csv", 
                       t, delimiter =", ",  # Set the delimiter as a comma followed by a space
                       fmt ='% s')
                       
                                # mask signal to -ve
                       mask_filt = i0.copy()
                       mask_filt[i0>0]=0 # negative
                       mask_abs = np.abs(mask_filt)
                       # find peaks in the signal
                       pks = find_peaks(mask_abs,distance=dist)[0]

                       if len(pks)==0:
                           raise ValueError(f"Failed to find -ve edge in Input 0!")
                       t = []
                       t = np.linspace(0, 5,len(i0[pks]))    
                       ax[0].plot(t,i0[pks],color=colours[ncolour])
                       
                       #max_negative_i0 = max(abs(i0[pks]))
                       #print("max_negative_i0") 
                       #print(-1*max_negative_i0)
                       
                       max_negative_i0_sum = sum(i0[pks]**2)
                    #   print("max_negative_i0_sum")
                    #   print(max_negative_i0_sum)
                       #np.savetxt(f"Stripe1_35gmin_Input_0_Data_i0_pks_50_499kkHz_Negative.csv", 
                       #i0[pks], delimiter =", ",  # Set the delimiter as a comma followed by a space
                       #fmt ='% s')
                       #np.savetxt(f"Stripe1_35gmin_Input_0_Data_t_50_499kHz_Negative.csv", 
                       #t, delimiter =", ",  # Set the delimiter as a comma followed by a space
                       #fmt ='% s')
 
                       #patches0 = [mpatches.Patch(color='blue', label="Signal"),mpatches.Patch(color='red', label="Edge")]
                       #ax[0].legend(handles=patches0)
                       #ax[0].set(xlabel="Time (s)",ylabel="Input 0")#,title=f"Input 0"+add_label)
                       #ax[0].set_title(f"{fname}, Input 0, {sname}, {modes_plot_string}, {freq_plot_string}Hz", fontsize=9)
                
                       ## find edge of Input 1
                       # mask signal to -ve
                       mask_filt = i1.copy()
                       mask_filt[i1<0]=0
                       # find peaks in the signal
                       pks = find_peaks(mask_filt,distance=dist)[0]
                       if len(pks)==0:
                           raise ValueError(f"Failed to find +ve edge in Input 1!")
                       # plot the signal and +ve edge
                       #ax[1].plot(time,i1,'b-')
                       t = []
                       t = np.linspace(0, 5,len(i1[pks]))
                       #print(t)
                       #print(len(i1[pks]))
                       ax[1].plot(t,i1[pks],color=colours[ncolour],label=feedrate[sname])
                       #max_positive_i1 = max(i1[pks])
                       #print("max_positive_i1")
                       #print(max_positive_i1)
                       
                       max_positive_i1_sum = sum(i1[pks]**2)
                    #   print("max_positive_i1_sum")
                    #   print(max_positive_i1_sum)
                       np.savetxt(f"Stripe5_35gmin_Input_1_Data_i1_pks_50_499kHz_Positive_Dist5e3.csv", 
                       i1[pks], delimiter =", ",  # Set the delimiter as a comma followed by a space
                       fmt ='% s')
                       np.savetxt(f"Stripe5_35gmin_Input_1_Data_t_50_499kHz_Positive_Dist5e3.csv", 
                       t, delimiter =", ",  # Set the delimiter as a comma followed by a space
                       fmt ='% s')
                       
                       #print(len(i1[pks])) 
                       # mask signal to +ve
                       mask_filt = i1.copy()
                       mask_filt[i1>0]=0
                       mask_abs = np.abs(mask_filt)
                       # find peaks in the signal
                       pks = find_peaks(mask_abs,distance=dist)[0]
                       
                       if len(pks)==0:
                           raise ValueError(f"Failed to find -ve edge in Input 1!")
                       t = []
                       t = np.linspace(0, 5,len(i0[pks]))  
                       ax[1].plot(t,i1[pks],color=colours[ncolour])
                       #max_negative_i1 = max(abs(i1[pks]))
                       #print("max_negative_i1")
                       #print(-1*max_negative_i1)
                       
                       max_negative_i1_sum = sum(i1[pks]**2)
                      # print("max_negative_i1_sum")
                      # print(max_negative_i1_sum)
                       
                       #np.savetxt(f"Stripe1_35gmin_Input_1_Data_i1_pks_100_250kHz_Negative.csv", 
                       #i1[pks], delimiter =", ",  # Set the delimiter as a comma followed by a space
                       #fmt ='% s')
                       #np.savetxt(f"Stripe1_35gmin_Input_1_Data_t_100_250kHz_Negative.csv", 
                       #t, delimiter =", ",  # Set the delimiter as a comma followed by a space
                       #fmt ='% s')
                       #Average = (abs(max_positive_i0) + abs(max_positive_i1) + abs(max_negative_i0) + abs(max_negative_i1))/4
                       #print("Average")
                       #print(Average)
                       Result_i0 = [max_positive_i0_sum + max_negative_i0_sum] 
                       Result_i1 = max_positive_i1_sum + max_negative_i1_sum
                       print(Result_i0, Result_i1)
                       ncolour+=1
           max_positive_i0 = np.append(max_positive_i0,max_positive_i0)
           max_negative_i0 = np.append(max_negative_i0,max_negative_i0)
           max_positive_i1 = np.append(max_positive_i1,max_positive_i1)
           max_negative_i1 = np.append(max_negative_i1,max_negative_i1)
                       
             
           #patches0 = [mpatches.Patch(color='blue', label="Signal"),mpatches.Patch(color='red', label="Edge")]
           #ax[0].legend(handles=patches0)
           ax[0].legend(fontsize=12)
           ax[0].set(xlabel="Time (s)",ylabel="Input 0")#,title=f"Input 0"+add_label)
           ax[0].set_title(f"{fname}, Input 0, {modes_plot_string}, {freq_plot_string}Hz", fontsize=9)
           ax[0].set_ylim([-0.085, 0.085])
           #patches1 = [mpatches.Patch(color='blue', label="Signal"),mpatches.Patch(color='red', label="Edge")]
           #ax[1].legend(handles=patches1)
           ax[1].legend(fontsize=12)
           ax[1].set(xlabel="Time (s)",ylabel="Input 1")#,title=f"Input 1"+add_label)
           ax[1].set_title(f"{fname}, Input 1, {modes_plot_string}, {freq_plot_string}Hz", fontsize=9)
           ax[1].set_ylim([-0.25, 0.25])
              #  return f
                #elif model == "overlay":
                   ## find edge of Input 0
                   # mask signal to -ve
                #   mask_filt = i0.copy()
                #   mask_filt[i0<0]=0
                #   # find peaks in the signal
                #   pks = find_peaks(mask_filt,distance=dist)[0]
                #   if len(pks)==0:
                #       raise ValueError(f"Failed to find +ve edge in Input 0!")

                #   f,ax = plt.subplots(constrained_layout=True)
                #   ax.plot(time[pks],i0[pks],'r-')
                    
                   # mask signal to -ve
                #   mask_filt = i0.copy()
                #   mask_filt[i0>0]=0
                #   mask_abs = np.abs(mask_filt)
                   # find peaks in the signal
                #   pks = find_peaks(mask_abs,distance=dist)[0]
                #   if len(pks)==0:
                #       raise ValueError(f"Failed to find -ve edge in Input 0!")

                #   ax.plot(time[pks],i0[pks],'r-')
                
                   ## find edge of Input 1
                   # mask signal to +ve
                #   mask_filt = i1.copy()
                #   mask_filt[i1<0]=0
                #   # find peaks in the signal
                #   pks = find_peaks(mask_filt,distance=dist)[0]
                #   if len(pks)==0:
                #       raise ValueError(f"Failed to find +ve edge in Input 1!")
                #   tax = ax.twinx()
                   # plot the signal and +ve edge
                #   tax.plot(time[pks],i1[pks],'b-')
                   
                   # mask signal to -ve
                #   mask_filt = i1.copy()
                #   mask_filt[i1>0]=0
                #   mask_abs = np.abs(mask_filt)
                #   # find peaks in the signal
                #   pks = find_peaks(mask_abs,distance=dist)[0]
                #   if len(pks)==0:
                #      raise ValueError(f"Failed to find -ve edge in Input 1!")

                #   tax.plot(time[pks],i1[pks],'b-')
                #    
                #  patches = [mpatches.Patch(color='red', label="Input 0"),mpatches.Patch(color='blue', label="Input 1")]
                #   ax.legend(handles=patches)
                #   ax.set(xlabel="Time (s)",ylabel="Input 0 Voltage (V)",title=f"{fname} {add_label}")
                   #ax.set_title(f"{fname}, Input 0, {sname}, {modes_plot_string}, {freq_plot_string}Hz", fontsize=9)
                #   tax.set_ylabel("Input 1 Voltage (V)")
                 #  f.suptitle(fname)
           #Results = [max_positive_i0_sum, max_negative_i0_sum, max_positive_i1_sum, max_negative_i1_sum]
           #Result_i0 = [max_positive_i0_sum + max_negative_i0_sum] 
           #Result_i1 = max_positive_i1_sum + max_negative_i1_sum
           #print(Result_i0, Result_i1)
           #np.savetxt(f"Maximum_signal_amplitude_AE_250_350kHz.csv", 
           #Results, delimiter =", ",  # Set the delimiter as a comma followed by a space
           #fmt ='% s') 
           #return f        



if __name__ == "__main__":
    import matplotlib.colors as colors
    from scipy.signal import periodogram
    plt.rcParams['agg.path.chunksize'] = 10000
    bandpassfilterOutline("ae/sheff_lsbu_stripe_coating_2.tdms", freq=[50e3, 499e3], mode=['highpass', 'lowpass'], dist=int(5e3),model="separate")