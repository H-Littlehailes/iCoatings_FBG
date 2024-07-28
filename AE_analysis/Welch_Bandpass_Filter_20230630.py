from nptdms import TdmsFile
import os
import seaborn as sns
import tempfile
import numpy as np
import csv
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
STRIPE_PERIOD_2 = {'sheff_lsbu_stripe_coating_2': {#'ref': [85, 90],
                                                 #'stripe_1_1':[107, 112],
                                                 #'stripe_2_1':[117.75, 122.75], #[118, 123],
                                                 #'stripe_3_1':[127.25, 133.25], #[127, 133],
                                                 #'stripe_4_1':[137, 143],
                                                 #'stripe_5_1':[146, 152], #[146, 152],
                                                # 2nd pass
                                                #'stripe_1_2':[157, 163], 
                                                #'stripe_2_2':[168.75, 174.75], #[169, 175],
                                                #'stripe_3_2':[180, 185.5], #[180, 185.5],
                                                #'stripe_4_2':[190.25, 195.25], #[190, 195],
                                                #'stripe_5_2':[199, 205], #[199.5, 205.5],
                                                # 3rd pass
                                                #'stripe_1_3':[214.5, 220.5], #[215, 221],
                                                #'stripe_2_3':[226.25, 231.25], #[226, 231], 
                                                #'stripe_3_3':[236.85, 242.85], #[237, 243],
                                                #'stripe_4_3':[249.5, 255.5],#[249, 255],
                                                #'stripe_5_3':[260, 266],
                                                 #unknown
                                                #'stripe_1_4':[276.75, 281.75], #[277, 282],
                                                #'stripe_2_4':[290.25, 296.25], #[290, 296],
                                                #'stripe_3_4':[304, 310],
                                                #'stripe_4_4':[314.75, 320.75], #[314, 320],
                                                'stripe_5_4':[326.75,332.75], #[327,333]
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
                                                #'stripe_5_1':'35 G/MIN',
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
                                                'stripe_5_4':'35 G/MIN (4)',
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

def welchStripes(path,**kwargs):
    '''
        Plot the Welch PSD of each stripe in the target file

        Each stripe is saved separately to the same location as the source file

        When freq_clip is used, the user controls the y axis limit by setting the yspace parameter.
        The y-axis lim is the max value within freq_clip + yspace x max value

        Inputs:
            path : TDMS fpath
            freq_clip : Frequency range to clip the plot to
            yspace : Fraction of max value to add to the top. Default 0.1.
    '''
    from scipy import signal
    sns.set_theme("paper")
    fname = os.path.splitext(path)[0]
    ftitle = os.path.basename(os.path.splitext(path)[0])
    # for each tdms
    for fn in glob(path):
        # check if supported
        periods = list(filter(lambda x : os.path.splitext(os.path.basename(fn))[0] in x,PERIODS))
        if periods:
            periods = periods[0][os.path.splitext(os.path.basename(fn))[0]]
            Pmax_i0 = 0
            Pmax_i1 = 0
            # filter to time periods dict
            for sname,chunk in periods.items():
                # load sub bit of the stripe
                time,i0,i1 = loadTDMSSub(path,chunk,is_time=True)
                # plot period
                fig0,ax0 = plt.subplots(constrained_layout=True)
                f, Pxx_den = signal.welch(i0, 1e6,nperseg=8192)
                ax0.plot(f,Pxx_den)
                
                joint0 = zip(f, Pxx_den)
                with open('Welch_PSD_Layer4_35gmin_Input_0_280_350kHz_nperseg_8192.csv', 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['f','Pxx_den'])
                    writer.writerows(joint0)
                
                
                ax0.set(xlabel="Frequency (Hz)",ylabel="PSD (V**2/Hz)",title=f"{ftitle}\nInput 0 {sname} Power Spectral Density")
                if kwargs.get("freq_clip",None):
                    f0,f1 = kwargs.get("freq_clip",None)
                    
                    ax0.set_xlim(f0,f1)
                    Pmax = Pxx_den[(f>=f0)&(f<=f1)].max()
                    #fx = np.arange(f0,f1, 4097)
                    ax0.set_ylim(0,Pmax+kwargs.get("yspace",0.10)*Pmax)
                    #np.savetxt('Welch_PSD_Layer1_15gmin_Input_0_280_350kHz_nperseg_8192.csv', np.vstack((fx,Pmax)).T, delimiter=', ')
                   # joint0 = zip(f, Pmax)
                   # with open('Welch_PSD_Layer1_20gmin_Input_0_280_350kHz_nperseg_8192.csv', 'wb+') as csvfile:
                   #     filewriter = csv.writer(csvfile)
                   #     filewriter.writerows(joint0)
                
               #     fig.savefig(fr"{fname}-Input 0-{sname}-welch-freq-clip-{f0}-{f1}.png")
               # else:
                #    fig.savefig(fr"{fname}-Input 0-{sname}-welch.png")
                #plt.close(fig)
        
                # plot period
                #fig1,ax1 = plt.subplots(constrained_layout=True)
                #f1, Pxx_den1 = signal.welch(i1, 1e6,nperseg=8192)
                #ax1.plot(f1,Pxx_den1)
                
                #joint1 = zip(f1, Pxx_den1)
                #with open('Welch_PSD_Layer2_15gmin_Input_1_280_350kHz_nperseg_8192.csv', 'w', newline='') as csvfile:
                #    writer = csv.writer(csvfile)
                #    writer.writerow(['f','Pxx_den'])
                #    writer.writerows(joint1)
                
                
                
                
                ax1.set(xlabel="Frequency (Hz)",ylabel="PSD (V**2/Hz)",title=f"{ftitle}\nInput 1 {sname} Power Spectral Density")
                if kwargs.get("freq_clip",None):
                    f0,f1 = kwargs.get("freq_clip",None)
                    ax1.set_xlim(f0,f1)
                    Pmax = Pxx_den[(f>=f0)&(f<=f1)].max()
                    ax1.set_ylim(0,Pmax+kwargs.get("yspace",0.10)*Pmax)
               #     np.savetxt('Welch_PSD_Layer1_15gmin_Input_1_280_350kHz_nperseg_8192.csv', np.vstack((f,Pxx_den)).T, delimiter=', ')
                #    fig.savefig(fr"{fname}-Input 1-{sname}-welch-freq-clip-{f0}-{f1}.png")
                #else:
                #    fig.savefig(fr"{fname}-Input 1-{sname}-welch.png")
                #plt.close(fig)
                return fig0, fig1
        else:
            print(f"Unsupported file {fn}!")
            
            
if __name__ == "__main__":
    import matplotlib.colors as colors
    from scipy.signal import periodogram
    plt.rcParams['agg.path.chunksize'] = 10000           
            
            
#welchStripes("ae/sheff_lsbu_stripe_coating_2.tdms",freq_clip=[280e3,350e3],use_fr=True)            
welchStripes("ae/sheff_lsbu_stripe_coating_1.tdms",freq_clip=[1e3,500e3],use_fr=True)            
