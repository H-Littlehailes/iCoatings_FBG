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
STRIPE_PERIOD_2 = {'sheff_lsbu_stripe_coating_2': {'ref': [85, 90],
                                                 #'stripe_1_1':[107, 112],
                                                 #'stripe_2_1':[118, 123],
                                                 #'stripe_3_1':[127, 133],
                                                 #'stripe_4_1':[137, 143],
                                                 #'stripe_5_1':[146, 152],
                                                # 2nd pass
                                                #'stripe_1_2':[157, 163], 
                                                #'stripe_2_2':[169, 175],
                                                #'stripe_3_2':[180, 185.5],
                                                #'stripe_4_2':[190, 195],
                                                #'stripe_5_2':[199, 205],
                                                # 3rd pass
                                                #'stripe_1_3':[215, 221],
                                                #'stripe_2_3':[226, 231], 
                                                #'stripe_3_3':[237, 243],
                                                #'stripe_4_3':[249, 255],
                                                #'stripe_5_3':[260, 266],
                                                 #unknown
                                                #'stripe_1_4':[277, 282],
                                                #'stripe_2_4':[290, 296],
                                                #'stripe_3_4':[304, 310],
                                                #'stripe_4_4':[314, 320],
                                                #'stripe_5_4':[327,333],
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

FEED_RATE_2 = {'sheff_lsbu_stripe_coating_2': {'ref'       : 'ref',
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
                                                #'stripe_5_4':'35 G/MIN (4)',
                                                 }}

FEED_RATE = [FEED_RATE_1, FEED_RATE_2]


## from https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
# band pass
def butter_bandpass(lowcut, highcut, order=5):
    nyq = 0.5 * 1e6
    low = lowcut / nyq
    high = highcut / nyq
    return butter(order, [low, high], btype='bandpass', analog=False, output='sos')

def butter_bandpass_filter(data, lowcut, highcut, order=6):
    sos = butter_bandpass(lowcut, highcut, order=order)
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
    # Find absolute value of the data and smooth
      #  data = abs(data)
    # plot both channels
    f0,ax0 = plt.subplots(constrained_layout=True,figsize=(9,8))
    sns.lineplot(data=data,x='Time (s)',y='Input 0',ax=ax0)
    ax0.set_title(f"{os.path.splitext(os.path.basename(path))[0]}, Input 0")
    
    f1,ax1 = plt.subplots(constrained_layout=True,figsize=(9,8))
    sns.lineplot(data=data,x='Time (s)',y='Input 1',ax=ax1)
    ax1.set_title(f"{os.path.splitext(os.path.basename(path))[0]}, Input 1")
    return f0,f1
    ##plt.show()  
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

def plotLombScargle(path,freqs=np.arange(300e3,350e3,1e3),normalize=True,**kwargs):
    '''
        Plot Lomb-Scargle periodgram of the given file between the target frequencies

        Input 0 and Input 1 are plotted on two separate figures and returned

        Inputs:
            path : TDMS path
            freqs : Array of frequencies to evaluate at
            normalize : Flag to normalize response
            tclip : 2-element time period to clip to
            input0_title : Figure title used on plot for Input 0
            input1_title : Figure title used on plot for Input 1

        Returns generated figures
    '''
    from scipy.signal import lombscargle
    time,i0,i1 = loadTDMSData(path)
    # convert freq to rad/s
    rads = freqs/(2*np.pi)
    # clip to time period if desired
    if not (kwargs.get("tclip",None) is None):
        tclip = kwargs.get("tclip",None)
        i0 = data[(time >= tclip[0]) & (time <= tclip[1])]
    # calculate
    pgram = lombscargle(i0, time, rads, normalize=normalize)
    f0,ax = plt.subplots(constrained_layout=True,figsize=(9,8))
    sns.lineplot(x=freqs,y=pgram,ax=ax)
    ax.set(xlabel="Frequency (rad/s)",ylabel="Normalized Amplitude",title="Input 0")
    f0.suptitle(kwargs.get("input0_title","Acoustic Lomb-Scargle Periodogram"))

    pgram = lombscargle(i1,time, rads, normalize=normalize)
    f1,ax = plt.subplots(constrained_layout=True,figsize=(9,8))
    sns.lineplot(x=freqs,y=pgram,ax=ax)
    ax.set(xlabel="Frequency (rad/s)",ylabel="Normalized Amplitude",title="Input 1")
    f1.suptitle(kwargs.get("input1_title","Acoustic Lomb-Scargle Periodogram"))
    return f0,f1

def plotSTFTSB(signal,nperseg=256,fclip=None,**kwargs):
    '''
        Plot the STFT of the target signal using Seaborn

        fclip is for clipping the freq to above the set threshold
        as contrast over the full range can be an issue

        Inputs:
            signal : Numpy array to perform stft on
            nperseg : Number of segments. See scipy.signal.stft
            fclip : Freq clipping threshold
            theme : Seaborn threshold to use

        Returns generated figure
    '''
    import pandas as pd
    sns.set_theme(kwargs.get("theme","paper"))
    f, t, Zxx = stft(signal, 1e6, nperseg=nperseg)
    if fclip:
        Zxx = Zxx[f>=fclip,:]
        f = f[f>=fclip]
    # convert data into dataframe
    data = pd.DataFrame(np.abs(Zxx),index=f,columns=t)
    ax = sns.heatmap(data)
    ax.invert_yaxis()
    return plt.gcf()

def plotSTFTPLT(signal,nperseg=256,fclip=None,use_log=True,**kwargs):
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
    import matplotlib.colors as colors
    # set seaborne theme
    sns.set_theme(kwargs.get("theme","paper"))
    # perform STFT at the set parameters
    f, t, Zxx = stft(signal, 1e6, nperseg=nperseg)
    # offset time vector for plotting
    t += kwargs.get("tmin",0.0)
    # if user wants the frequency clipped above a particular point
    if fclip:
        Zxx = Zxx[f>=fclip,:]
        f = f[f>=fclip]
    fig,ax = plt.subplots(constrained_layout=True)
    # if using colormap log Norm
    if use_log:
        X = ax.pcolormesh(t, f, np.abs(Zxx), norm=colors.LogNorm(vmin=np.abs(Zxx).max()*0.01, vmax=np.abs(Zxx).max()),cmap=kwargs.get("cmap",'inferno'),shading=kwargs.get("shading",'gouraud'))
    else:
        X = ax.pcolormesh(t, f, np.abs(Zxx),cmap=kwargs.get("cmap",'inferno'),shading=kwargs.get("shading",'gouraud'))
    # set title
    ax.set_title(kwargs.get("title",'STFT Magnitude'))
    ax.set_ylabel('Frequency [Hz]')
    ax.set_xlabel('Time [sec]')
    # set colorbar
    plt.colorbar(X)
    return fig

def STFTWithLombScargle(path,span=0.4,n_bins=100, grid_size=100,min_freq=300e3,max_freq=400e3):
    # see https://stackoverflow.com/a/65843574
    import fitwrap as fw

    if isinstance(path,str):
        data = loadTDMSData(path)
    else:
        data = path
    # extract columns
    signaldata = data.values[:,0]
    t = data.values[:,-1]
    # time bins
    x_bins = np.linspace(t.min()+span, t.max()-span, n_bins)
    # area for spectrogram
    spectrogram = np.zeros([grid_size, x_bins.shape[0]])
    # iterate over bins
    for index, x_bin in enumerate(x_bins):
        # build mask to isolate sequence
        mask = np.logical_and((x_bin-span)<=t, (x_bin+span)>=t)
        # perform periodigram over period
        frequency_grid, lombscargle_spectrum = fw.lomb_spectrum(t[mask], signaldata[mask],
                        frequency_span=[min_freq, max_freq], grid_size=grid_size)
        # update spectrogram using results
        spectrogram[:, index] = lombscargle_spectrum

    # plot results
    plt.imshow(spectrogram, aspect='auto', extent=[x_bins[0],x_bins[-1],
                frequency_grid[0],frequency_grid[-1]], origin='lower') 
    plt.xlabel('Time in seconds')
    plt.ylabel('Frequency')
    return plt.gcf()

def plotSignalEnergyFreqBands(path,fstep,tclip=None,fmin=0.1):
    '''
        Plot the signal energy within frequency bands

        Going from fmin to 1e6/2 in steps of fstep
        eg. [0,fstep],[fstep,2*step] etc.

        Frequencies are limited to these ranges using bandpass Butterworth filter

        tclip is for specifying a specific time period to look at. Useful for targeting
        a specifc stripe or for at least limiting it to the activity to manage memory better

        Inputs:
            data : String or result of loadTDMSData or loadTDMSSub
            fstep : Frequency steps
            tclip : Time period to clip to. Default None
            fmin : Min frequency to start at. Has to be non-zero. Default 0.1 Hz

        Return generated figure
    '''
    from scipy.fft import rfft, irfft, rfftfreq

    if isinstance(path,str):
        data = loadTDMSData(path)
    if tclip:
        data = data[(data['Time (s)'] >= tclip[0]) & (data['Time (s)'] <= tclip[1])]
    fsteps = np.arange(fmin,1e6/2,fstep)
    print(fsteps.shape[0])
    energy = {'Input 0':[],'Input 1':[]}
    # make an axis for each signal
    fig,ax = plt.subplots(ncols=data.shape[1]-1,constrained_layout=True,figsize=(7*data.shape[1]-1,6))
    for c in ['Input 0','Input 1']:
        signal = data[c]
        # for each freq band
        for fA,fB in zip(fsteps,fsteps[1:]):
            yf = butter_bandpass_filter(signal,fA,fB,order=6 )
            energy[c].append(np.sum(yf**2))
    # calculate the bar locations
    locs = [fA+fstep/2 for fA in fsteps[:-1]]
    for aa,(cc,ee) in zip(ax,energy.items()):
        aa.bar(locs,ee,width=fstep,align='center')
    return fig

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
    '''
    # for each tdms
    for fn in glob(path):
        periods = list(filter(lambda x : os.path.splitext(os.path.basename(fn))[0] in x,PERIODS))
        if periods:
            periods = periods[0][os.path.splitext(os.path.basename(fn))[0]]
            # filter to time periods dict
            for sname,chunk in periods.items():
                time,i0,i1 = loadTDMSSub(path,chunk,is_time=True)
                # perform stft for each channel
                f=plotSTFTPLT(i0,nperseg,fclip,use_log,title=f"{os.path.splitext(os.path.basename(fn))[0]}, Input 0, {sname} STFT",tmin=chunk[0],**kwargs)
                if fclip:
                    f.savefig(fr"{os.path.splitext(fn)[0]}-Input 0-stft-stripe-{sname}-clip-{fclip}.png")
                else:
                    f.savefig(fr"{os.path.splitext(fn)[0]}-Input 0-stft-stripe-{sname}.png")
                plt.close(f)

                # perform stft for each channel
                f=plotSTFTPLT(i1,nperseg,fclip,use_log,title=f"{os.path.splitext(os.path.basename(fn))[0]}, Input 1, {sname} STFT",tmin=chunk[0],**kwargs)
                if fclip:
                    f.savefig(fr"{os.path.splitext(fn)[0]}-Input 1-stft-stripe-{sname}-clip-{fclip}.png")
                else:
                    f.savefig(fr"{os.path.splitext(fn)[0]}-Input 1-stft-stripe-{sname}.png")
                plt.close(f)
        else:
            print(f"Unsupported file {fn}!")

def plotStripes(path,ds=None,**kwargs):
    '''
        Plot the STFT of each stripe in each channel

        This relies on the stripe time periods being listed in STRIPE_PERIOD and the file being supported

        fclip is the lower frequency to threshold above. This is to avoid the noise floor of signals and focus non the interesting stuff.

        When use_log is True, matplotlib.colors.LogNorm is being used

        Figures are saved to the same location at the source file

        Input:
            path : TDMS path
            ds : Rate of downsampling
    '''
    sns.set_theme("paper")
    # for each tdms
    for fn in glob(path):
        # check if supported
        periods = list(filter(lambda x : os.path.splitext(os.path.basename(fn))[0] in x,PERIODS))
        if periods:
            periods = periods[0][os.path.splitext(os.path.basename(fn))[0]]
            # filter to time periods dict
            for sname,chunk in periods.items():
                # load sub bit of the stripe
                time,i0,i1 = loadTDMSSub(path,chunk,is_time=True)
                # plot period
                f,ax = plt.subplots(constrained_layout=True,figsize=(9,8))
                ax.plot(time,i0)
                ax.set(xlabel="Time (s)",ylabel="Voltage (V)",title="Input 0")
                # set title
                ax.set_title(f"{os.path.splitext(os.path.basename(fn))[0]}, Input 0, {sname}")
                ax.figure.savefig(fr"{os.path.splitext(fn)[0]}-Input 0-time-stripe-{sname}.png")
                plt.close(f)

                # plot period
                f,ax = plt.subplots(constrained_layout=True,figsize=(9,8))
                ax.plot(time,i1)
                ax.set(xlabel="Time (s)",ylabel="Voltage (V)",title="Input 1")
                # set title
                ax.set_title(f"{os.path.splitext(os.path.basename(fn))[0]}, Input 1, {sname}")
                ax.figure.savefig(fr"{os.path.splitext(fn)[0]}-Input 1-time-stripe-{sname}.png")
                plt.close(f)
        else:
            print(f"Unsupported file {fn}!")

def plotStripesLimits(path,**kwargs):
    '''
        Plot the max value of each stripe for each tool.

        This relies on the stripe time periods being listed in STRIPE_PERIOD and the file being supported.

        A filter can be applied to the signal using the mode, order and and cutoff_freq keywords.
        The cutoff_freq keyword is the cutoff frequency of the filter.
        Can either be:
            - Single value representing a highpass/lowpass filter
            - 2-element Tuple/list for a bandpass filter.
            - List of values or list/tuples for a series of filters applied sequentially

        The mode parameter is to specify whether the filters are lowpass ("lp") or highpass ("hp"). If it's a single string, then it's applied
        to all non-bandpass filters in cutoff_freq. If it's a list, then it MUST be the same length as the number of filters

        The max value of each stripe is stored and plotted on a set of axis.

        Inputs:
            path : TDMS file path
            cutoff_freq : Float or tuple/list representing a cutoff frequency. Default None.
            mode : String stating whether it's a lowpass or highpass respectively. Default lp.
            order : Butterworth model order. Default 10.

        Returns a plot with max signal data
    '''
    import matplotlib.patches as mpatches
    from matplotlib.ticker import MaxNLocator
    mode_dict = {'lp': "Low Pass",'hp':"High Pass",'bp':"Bandpass","lowpass":"Low Pass","highpass":"High Pass","bandpass":"Bandpass"}
    fname = os.path.splitext(os.path.basename(path))[0]
    # check if supported
    periods = list(filter(lambda x : fname in x,PERIODS))
    if periods:
        periods = periods[0][fname]
        i0_max = []
        i0_min = []
        i1_max = []
        i1_min = []
        if kwargs.get("cutoff_freq",None):
            cf = kwargs.get("cutoff_freq",None)
            # if it's a single value, ensure it's a list
            if isinstance(cf,(int,float)):
                cf = [cf,]
            # if mode is a single string them treat all filters as that mode
            if isinstance(kwargs.get("mode","lowpass"),str):
                modes = len(cf)*[kwargs["mode"],]
            elif len(cf) != len(kwargs.get("mode","lowpass")):
                raise ValueError("Number of modes must match the number of filters!")
            else:
                modes = kwargs.get("mode","lowpass")
        # filter to time periods dict
        for sname,chunk in periods.items():
            # load sub bit of the stripe
            _,i0,i1 = loadTDMSSub(path,chunk,is_time=True)
            # iterate over each filter and mode
            for c,m in zip(cf,modes):
                print(c,m)
                # if it's a single value then it's a highpass/lowpass filter
                if isinstance(c,(int,float)):
                    sos = butter(kwargs.get("order",10), c/(1e6/2), m, fs=1e6, output='sos',analog=False)
                    i0 = sosfilt(sos, i0)
                    i1 = sosfilt(sos, i1)
                # if it's a list/tuple then it's a bandpass filter
                elif isinstance(c,(tuple,list)):
                    if m == "bandpass":
                        i0 = butter_bandpass_filter(i0,c[0],c[1],kwargs.get("order",10))
                        i1 = butter_bandpass_filter(i1,c[0],c[1],kwargs.get("order",10))
                    elif m == "bandstop":
                        i0 = butter_bandstop_filter(i0,c[0],c[1],kwargs.get("order",10))
                        i1 = butter_bandstop_filter(i1,c[0],c[1],kwargs.get("order",10))
            i0_max.append(i0.max())
            i0_min.append(i0.min())

            i1_max.append(i1.max())
            i1_min.append(i1.min())
        # combine modes together
        modes_save_string = '-'.join(modes)
        modes_plot_string = ','.join(modes)
        freq_save_string = '-'.join([str(c) for c in cf])
        freq_plot_string = ','.join([str(c) for c in cf])
        # get number of values
        nf = len(i0_max)
        # get plotting mode
        pmode = kwargs.get("plot_mode","both")
        # plot both min and max signal amplitudes
        if pmode == "both":
            f,ax = plt.subplots(ncols=2,constrained_layout=True,figsize=(16,8))
            # ensure that the axis ticks are integer
            ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
            ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
            # plot max
            ax[0].plot(range(nf),i0_max,'r-')
            ax[0].set_xticks(range(nf))
            # make twin axis for min
            tax = ax[0].twinx()
            # plot min
            tax.plot(range(nf),i0_min,'b-')
            # set labels based on if the user gave a cutoff freq
            if kwargs.get("cutoff_freq",None) is None:
                ax[0].set(xlabel="Stripe",ylabel="Max Voltage (V)",title="Input 0")
                ax[1].set(xlabel="Stripe",ylabel="Max Voltage (V)",title="Input 1")
            else:
                ax[0].set(xlabel="Stripe",ylabel="Max Voltage (V)",title=f"Input 0, {modes_plot_string} {freq_plot_string}Hz")
                ax[1].set(xlabel="Stripe",ylabel="Max Voltage (V)",title=f"Input 1, {modes_plot_string} {freq_plot_string}Hz")
            tax.set_ylabel("Min Voltage (V)")
            # make patches for legend
            patches = [mpatches.Patch(color='blue', label="Min Voltage"),mpatches.Patch(color='red', label="Max Voltage")]
            ax[0].legend(handles=patches)
            # plot Input 1 max
            ax[1].plot(range(nf),i1_max,'r-')
            ax[1].set_xticks(range(nf))
            tax = ax[1].twinx()
            # plot Input 1 min
            tax.plot(range(nf),i1_min,'b-')
            tax.set_ylabel("Min Voltage (V)")
            # make patches for ax[1]
            patches = [mpatches.Patch(color='blue', label="Min Voltage"),mpatches.Patch(color='red', label="Max Voltage")]
            ax[1].legend(handles=patches)
            # set title to the filename
            f.suptitle(fname)
            # save figure to the same location s the source file with the filename containing cutoff freq
            if kwargs.get("cutoff_freq",None):
                f.savefig(fr"{os.path.splitext(path)[0]}-signal-limits-filtered-{modes_save_string}-freq-{freq_save_string}.png")
            else:
                f.savefig(fr"{os.path.splitext(path)[0]}-signal-limits.png")
        # plot ONLY max
        elif pmode == "max":
            f,ax = plt.subplots(ncols=2,constrained_layout=True,figsize=(16,8))
            # ensure that the axis ticks are integer
            ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
            ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
            ax[0].plot(range(nf),i0_max,'r-')
            ax[1].plot(range(nf),i1_max,'r-')
            ax[0].set_xticks(range(nf))
            ax[1].set_xticks(range(nf))
            if kwargs.get("cutoff_freq",None) is None:
                ax[0].set(xlabel="Stripe",ylabel="Max Voltage (V)",title="Input 0")
                ax[1].set(xlabel="Stripe",ylabel="Max Voltage (V)",title="Input 1")
            else:
                ax[0].set(xlabel="Stripe",ylabel="Max Voltage (V)",title=f"Input 0, {modes_plot_string} {freq_plot_string}Hz")
                ax[1].set(xlabel="Stripe",ylabel="Max Voltage (V)",title=f"Input 1, {modes_plot_string} {freq_plot_string}Hz")
            # set title to the filename
            f.suptitle(fname)
            # save figure to the same location s the source file with the filename containing cutoff freq
            if kwargs.get("cutoff_freq",None):
                f.savefig(fr"{os.path.splitext(path)[0]}-signal-limits-filtered-{modes_save_string}-freq-{freq_save_string}-max-only.png")
            else:
                f.savefig(fr"{os.path.splitext(path)[0]}-signal-limits-max-only.png")
        # plot ONLY min
        elif pmode == "min":
            f,ax = plt.subplots(ncols=2,constrained_layout=True,figsize=(14,6))
            ax[0].xaxis.set_major_locator(MaxNLocator(integer=True))
            ax[1].xaxis.set_major_locator(MaxNLocator(integer=True))
            ax[0].plot(range(nf),i0_min,'b-')
            ax[1].plot(range(nf),i1_min,'b-')
            ax[0].set_xticks(range(nf))
            ax[1].set_xticks(range(nf))
            if kwargs.get("cutoff_freq",None) is None:
                ax[0].set(xlabel="Stripe",ylabel="Min Voltage (V)",title="Input 0")
                ax[1].set(xlabel="Stripe",ylabel="Min Voltage (V)",title="Input 1")
            else:
                ax[0].set(xlabel="Stripe",ylabel="Min Voltage (V)",title=f"Input 0, {modes_plot_string} {freq_plot_string}Hz")
                ax[1].set(xlabel="Stripe",ylabel="Min Voltage (V)",title=f"Input 1, {modes_plot_string} {freq_plot_string}Hz")
            # set title to the filename
            f.suptitle(fname)
            # save figure to the same location s the source file with the filename containing cutoff freq
            #if kwargs.get("cutoff_freq",None):
            #    f.savefig(fr"{os.path.splitext(path)[0]}-signal-limits-filtered-{modes_save_string}-freq-{freq_save_string}-min-only.png")
            #else:
            #    f.savefig(fr"{os.path.splitext(path)[0]}-signal-limits-min-only.png")
            #plt.close(f)
        return f
    else:
        print(f"Unsupported file {fn}!")

def drawEdgeAroundStripe(path,dist=int(50e3),mode="separate",**kwargs):
    '''
        Draw around the edge of the stripe AE data using the peaks

        The +ve and -ve edges of Input 0 and Input 1 are identified and plotted
        The idea is to get a sense of how the area of each input differs over the same time period

        The mode input controls how the edges are plotted
            separate : Plot the signal and edges on separate axis
            overlay : Plot the edges on the same axis with NO signal

        This is designed for a SINGLE stripe as opposed to all of them

        The dist parameter is the min distance between peaks and is passed to find_peaks. Treat
        like a smoothing paramter of sorts

        Inputs:
            path : Path to TDMS file
            dist : Distance between peaks. See scipy.signal.find_peaks
            mode : Plotting mode. Default separate.
            
            Method of defining what stripe to process (see loadTDMSSub)
                time_period : Two-element iterable of time period to look at
                index_period: Two-element iterable of index period to look at
                stripe_ref : String or index reference of stripe.

        Returns generated figure
    '''
    from scipy.signal import find_peaks
    import matplotlib.patches as mpatches
    fname = os.path.splitext(os.path.basename(path))[0]
    add_label = ""
    # check how the user is identifying the stripe
    if kwargs.get("time_period",None):
        time,i0,i1 = loadTDMSSub(path,kwargs.get("time_period",None),is_time=True)
        add_label=f" Time={kwargs.get('time_period',None)}"
    elif kwargs.get("index_period",None):
        time,i0,i1 = loadTDMSSub(path,kwargs.get("index_period",None),is_time=False)
        
    elif "stripe_ref" in kwargs:
        periods = list(filter(lambda x : fname in x,PERIODS))
        if len(periods)==0:
            raise ValueError(f"Path {path} has no supported periods!")
        periods = periods[0][fname]
        # if referencing
        if isinstance(kwargs.get("stripe_ref",None),str):
            if not (kwargs.get("stripe_ref",None) in periods):
                raise ValueError(f"Stripe reference {kwargs.get('stripe_ref',None)} is invalid!")
            ref = kwargs.get("stripe_ref",None)
        elif isinstance(kwargs.get("stripe_ref",None),int):
            if kwargs.get("stripe_ref",None) > len(periods):
                raise ValueError(f"Out of bounds stripe index!")
            ref = list(periods.keys())[int(kwargs.get("stripe_ref"))]
        chunk = periods[ref]
        add_label=f" Stripe {ref}"
        # load time period
        time,i0,i1 = loadTDMSSub(path,chunk,is_time=True)
    else:
        raise ValueError("Missing reference to target period!")

    if mode == "separate":
        ## find edge of Input 0
        # mask signal to +ve
        mask_filt = i0.copy()
        mask_filt[i0<0]=0
        # find peaks in the signal
        pks = find_peaks(mask_filt,distance=dist)[0]
        if len(pks)==0:
            raise ValueError(f"Failed to find +ve edge in Input 0!")
        # plot the signal and +ve edge
        f,ax = plt.subplots(ncols=2,constrained_layout=True,figsize=(14,6))
        #ax[0].plot(time,i0,'b-')
        ax[0].plot(time[pks],i0[pks],'r-')
        
        # mask signal to -ve
        mask_filt = i0.copy()
        mask_filt[i0>0]=0
        mask_abs = np.abs(mask_filt)
        # find peaks in the signal
        pks = find_peaks(mask_abs,distance=dist)[0]

        if len(pks)==0:
            raise ValueError(f"Failed to find -ve edge in Input 0!")
        ax[0].plot(time[pks],i0[pks],'r-')

        patches = [mpatches.Patch(color='blue', label="Signal"),mpatches.Patch(color='red', label="Edge")]
        ax[0].legend(handles=patches)
        ax[0].set(xlabel="Time (s)",ylabel="Input 0",title=f"Input 0"+add_label)

        ## find edge of Input 1
        # mask signal to +ve
        mask_filt = i1.copy()
        mask_filt[i1<0]=0
        # find peaks in the signal
        pks = find_peaks(mask_filt,distance=dist)[0]
        if len(pks)==0:
            raise ValueError(f"Failed to find +ve edge in Input 1!")
        # plot the signal and +ve edge
        #ax[1].plot(time,i1,'b-')
        ax[1].plot(time[pks],i1[pks],'r-')
        
        # mask signal to +ve
        mask_filt = i1.copy()
        mask_filt[i1>0]=0
        mask_abs = np.abs(mask_filt)
        # find peaks in the signal
        pks = find_peaks(mask_abs,distance=dist)[0]

        if len(pks)==0:
            raise ValueError(f"Failed to find -ve edge in Input 1!")
        ax[1].plot(time[pks],i1[pks],'r-')

        patches = [mpatches.Patch(color='blue', label="Signal"),mpatches.Patch(color='red', label="Edge")]
        ax[1].legend(handles=patches)
        ax[1].set(xlabel="Time (s)",ylabel="Input 0",title=f"Input 1"+add_label)
    elif mode == "overlay":
        ## find edge of Input 0
        # mask signal to +ve
        mask_filt = i0.copy()
        mask_filt[i0<0]=0
        # find peaks in the signal
        pks = find_peaks(mask_filt,distance=dist)[0]
        if len(pks)==0:
            raise ValueError(f"Failed to find +ve edge in Input 0!")

        f,ax = plt.subplots(constrained_layout=True)
        ax.plot(time[pks],i0[pks],'r-')
        
        # mask signal to -ve
        mask_filt = i0.copy()
        mask_filt[i0>0]=0
        mask_abs = np.abs(mask_filt)
        # find peaks in the signal
        pks = find_peaks(mask_abs,distance=dist)[0]
        if len(pks)==0:
            raise ValueError(f"Failed to find -ve edge in Input 0!")

        ax.plot(time[pks],i0[pks],'r-')

        ## find edge of Input 1
        # mask signal to +ve
        mask_filt = i1.copy()
        mask_filt[i1<0]=0
        # find peaks in the signal
        pks = find_peaks(mask_filt,distance=dist)[0]
        if len(pks)==0:
            raise ValueError(f"Failed to find +ve edge in Input 1!")
        tax = ax.twinx()
        # plot the signal and +ve edge
        tax.plot(time[pks],i1[pks],'b-')
        
        # mask signal to -ve
        mask_filt = i1.copy()
        mask_filt[i1>0]=0
        mask_abs = np.abs(mask_filt)
        # find peaks in the signal
        pks = find_peaks(mask_abs,distance=dist)[0]
        if len(pks)==0:
            raise ValueError(f"Failed to find -ve edge in Input 1!")

        tax.plot(time[pks],i1[pks],'b-')

        patches = [mpatches.Patch(color='red', label="Input 0"),mpatches.Patch(color='blue', label="Input 1")]
        ax.legend(handles=patches)
        ax.set(xlabel="Time (s)",ylabel="Input 0 Voltage (V)",title=f"{fname} {add_label}")
        tax.set_ylabel("Input 1 Voltage (V)")
    f.suptitle(fname)
    return f

def plotAllStripeEdges(path,dist=int(50e3),**kwargs):
    '''
        Trace the edges of each stripe and draw the edges for Input 0 and Input 1 on the same axis

        The edges are traced using find_peaks where the user sets the dist parameter.

        It is recommended that dist be an important frequency like 50e3 as it traces over the strongest
        frequencies well. Increasing it acts as a smoothing parameter whilst decreasing means it picks up the valleys
        more

        All stripe edges are drawn on the SAME axis

        Inputs:
            path : TDMS file path
            dist : Distance between peaks. See find_peaks

        Returns figure
    '''
    from scipy.signal import find_peaks
    import matplotlib.patches as mpatches
    from matplotlib.pyplot import cm
    fname = os.path.splitext(os.path.basename(path))[0]
    add_label = ""
    # check if supported
    periods = list(filter(lambda x : os.path.splitext(os.path.basename(path))[0] in x,PERIODS))
    if periods:
        periods = periods[0][os.path.splitext(os.path.basename(path))[0]]
        f,ax = plt.subplots(ncols=2,constrained_layout=True)
        color = cm.rainbow(np.linspace(0, 1, len(periods)))
        # filter to time periods dict
        for (sname,chunk),c in zip(periods.items(),color):
            time,i0,i1 = loadTDMSSub(path,chunk,is_time=True)
            time = (time-time.min())/(time.max()-time.max())
            ## find edge of Input 0
            # mask signal to +ve
            mask_filt = i0.copy()
            mask_filt[i0<0]=0
            # find peaks in the signal
            pks = find_peaks(mask_filt,distance=dist)[0]
            if len(pks)==0:
                raise ValueError(f"Failed to find +ve edge in Input 0!")

            ax[0].plot(time[pks],i0[pks],c)
            
            # mask signal to -ve
            mask_filt = i0.copy()
            mask_filt[i0>0]=0
            mask_abs = np.abs(mask_filt)
            # find peaks in the signal
            pks = find_peaks(mask_abs,distance=dist)[0]
            if len(pks)==0:
                raise ValueError(f"Failed to find -ve edge in Input 0!")

            ax[0].plot(time[pks],i0[pks],c)

            ## find edge of Input 1
            # mask signal to +ve
            mask_filt = i1.copy()
            mask_filt[i1<0]=0
            # find peaks in the signal
            pks = find_peaks(mask_filt,distance=dist)[0]
            if len(pks)==0:
                raise ValueError(f"Failed to find +ve edge in Input 1!")
            # plot the signal and +ve edge
            ax[1].plot(time[pks],i1[pks],c)
            
            # mask signal to -ve
            mask_filt = i1.copy()
            mask_filt[i1>0]=0
            mask_abs = np.abs(mask_filt)
            # find peaks in the signal
            pks = find_peaks(mask_abs,distance=dist)[0]
            if len(pks)==0:
                raise ValueError(f"Failed to find -ve edge in Input 1!")

            ax[1].plot(time[pks],i1[pks],c)

        patches = [mpatches.Patch(color=c, label=sname) for sname,c in zip(periods.keys(),color)]
        ax[0].legend(handles=patches)
        ax[1].legend(handles=patches)
        ax[0].set(xlabel="Time (s)",ylabel="Input 0 Voltage (V)",title="Input 0")
        ax[1].set(xlabel="Time (s)",ylabel="Input 1 Voltage (V)",title="Input 1")
        return f

# from https://stackoverflow.com/a/30408825
# shoelace formula
def PolyArea(x,y):
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))

def calcStripeAreas(path,dist=int(50e3),**kwargs):
    '''
        Find the edges of the stripe and estimate the area treating the edges as a complicated polygon

        The edges are traced using find_peaks where the user sets the dist parameter.

        It is recommended that dist be an important frequency like 50e3 as it traces over the strongest
        frequencies well. Increasing it acts as a smoothing parameter whilst decreasing means it picks up the valleys
        more

        All stripe edges are drawn on the SAME axis.

        The area of each stripe for Input 0 and Input 1 is plotted on separate exes

        Inputs:
            path : TDMS path
            dist : Distance between peaks. See find_peaks

        Returns figure
    '''
    from scipy.signal import find_peaks
    import matplotlib.patches as mpatches
    from matplotlib.pyplot import cm
    sns.set_theme("paper")
    fname = os.path.splitext(os.path.basename(path))[0]
    add_label = ""
    # check if supported
    periods = list(filter(lambda x : os.path.splitext(os.path.basename(path))[0] in x,PERIODS))
    feedrate = list(filter(lambda x : os.path.splitext(os.path.basename(path))[0] in x,FEED_RATE))
    if feedrate:
        feedrate = feedrate[0][os.path.splitext(os.path.basename(path))[0]] #Select Feedrate base on the source file
    if periods:
        periods = periods[0][os.path.splitext(os.path.basename(path))[0]]
        
        color = cm.rainbow(np.linspace(0, 1, len(periods)))
        time_pts_i0 = []
        v_pts_i0 = []
        area_i0 = []

        time_pts_i1 = []
        v_pts_i1 = []
        area_i1 = []
        # filter to time periods dict
        for (sname,chunk),c in zip(periods.items(),color):
            time,i0,i1 = loadTDMSSub(path,chunk,is_time=True)
            ## find edge of Input 0
            # mask signal to +ve
            mask_filt = i0.copy()
            mask_filt[i0<0]=0
            # find peaks in the signal
            pks = find_peaks(mask_filt,distance=dist)[0]
            if len(pks)==0:
                raise ValueError(f"Failed to find +ve edge in Input 0!")

            time_pts_i0.extend(time[pks].tolist())
            v_pts_i0.extend(i0[pks].tolist())
            
            # mask signal to -ve
            mask_filt = i0.copy()
            mask_filt[i0>0]=0
            mask_abs = np.abs(mask_filt)
            # find peaks in the signal
            pks = find_peaks(mask_abs,distance=dist)[0]
            if len(pks)==0:
                raise ValueError(f"Failed to find -ve edge in Input 0!")

            # add the points so that it's in clockwise order
            time_pts_i0.extend(time[pks].tolist()[::-1])
            v_pts_i0.extend(i0[pks].tolist()[::-1])
            area_i0.append(PolyArea(time_pts_i0,v_pts_i0))
            
           # f0,ax0 = plt.subplots(constrained_layout=True,figsize=(9,8))
            #plt.plot(time_pts_i0,v_pts_i0, 'r-')
            #ax0.set_title(f"{os.path.splitext(os.path.basename(path))[0]}, Input 0")
            
            ## find edge of Input 1
            # mask signal to +ve
            mask_filt = i1.copy()
            mask_filt[i1<0]=0
            # find peaks in the signal
            pks = find_peaks(mask_filt,distance=dist)[0]
            if len(pks)==0:
                raise ValueError(f"Failed to find +ve edge in Input 1!")

            time_pts_i1.extend(time[pks].tolist())
            v_pts_i1.extend(i0[pks].tolist())

            # mask signal to -ve
            mask_filt = i1.copy()
            mask_filt[i1>0]=0
            mask_abs = np.abs(mask_filt)
            # find peaks in the signal
            pks = find_peaks(mask_abs,distance=dist)[0]
            if len(pks)==0:
                raise ValueError(f"Failed to find -ve edge in Input 1!")

            time_pts_i1.extend(time[pks].tolist()[::-1])
            v_pts_i1.extend(i0[pks].tolist()[::-1])
            area_i1.append(PolyArea(time_pts_i1,v_pts_i1))

            

            #f1,ax1 = plt.subplots(constrained_layout=True,figsize=(9,8))
            #plt.plot(time_pts_i1,v_pts_i1, 'b--')
            #ax1.set_title(f"{os.path.splitext(os.path.basename(path))[0]}, Input 1")
            #return f0,f1


            time_pts_i0.clear()
            v_pts_i0.clear()
            
            time_pts_i1.clear()
            v_pts_i1.clear()
        # plot
        f,ax = plt.subplots(ncols=2,constrained_layout=True)
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
        f.suptitle(os.path.splitext(os.path.basename(path))[0]+" Shoelace Area between Edges")
        return f 
        #return f0,f1
    
    
    
    

def filterStripes(fn,freq=100e3,mode='bandpass',**kwargs):  #(fn,freq=50e3,mode='highpass',**kwargs)
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
    periods = list(filter(lambda x : fname in x,PERIODS))
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
            for c,m in zip(freq,modes):
                print(sname,c,m)
                # if it's a single value then it's a highpass/lowpass filter
                if isinstance(c,(int,float)):
                    print("bing!")
                    sos = butter(kwargs.get("order",10), c, m, fs=1e6, output='sos',analog=False)
                    i0 = sosfilt(sos, i0)
                    i1 = sosfilt(sos, i1)
                # if it's a list/tuple then it's a bandpass filter
                elif isinstance(c,(tuple,list)):
                    print("bong!")
                    if m == "bandpass":
                        i0 = butter_bandpass_filter(i0,c[0],c[1],kwargs.get("order",10))
                        i1 = butter_bandpass_filter(i1,c[0],c[1],kwargs.get("order",10))
                    elif m == "bandstop":
                        i0 = butter_bandstop_filter(i0,c[0],c[1],kwargs.get("order",10))
                        i1 = butter_bandstop_filter(i1,c[0],c[1],kwargs.get("order",10))
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
            
            return f_i0, f_i1
    else:
        print(f"Unsupported file {fn}!")

def bandpassfilterStripes(fn,freq = [100e3, 250e3],mode=['highpass','lowpass'],**kwargs):  #(fn,freq=50e3,mode='highpass',**kwargs)
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
            
            return f_i0, f_i1
    else:
        print(f"Unsupported file {fn}!")

def stripeSpectrogram(fn,shift=False):
    '''
        Plot the spectrogram of each stripe in the target file

        Each file is 

        Inputs:
            fn : TDMS path
    '''
    from scipy import signal
    from scipy.signal import spectrogram
    import matplotlib.colors as colors
    #from matplotlib.colors import LogNorm
    fname = os.path.splitext(os.path.basename(fn))[0]
    # check if supported
    periods = list(filter(lambda x : fname in x,PERIODS))
    if periods:
        periods = periods[0][fname]
        # filter to time periods dict
        for sname,chunk in periods.items():
            # load sub bit of the stripe
            time,i0,i1 = loadTDMSSub(fn,chunk,is_time=True)

            f,t,Zxx = signal.spectrogram(i0,1e6)
            fig,ax = plt.subplots()
            if shift:
                #plt.colorbar(ax.pcolormesh(t, np.fft.fftshift(f), np.fft.fftshift(Zxx, axes=0)**2),norm=colors.LogNorm())
                plt.colorbar(ax.pcolormesh(t, np.fft.fftshift(f), np.fft.fftshift(10*np.log10(Zxx), axes=0)), norm=colors.LogNorm())
            else:
                #plt.colorbar(ax.pcolormesh(t, f, Zxx ** 2),norm=colors.LogNorm())
                plt.colorbar(ax.pcolormesh(t, f, np.log10(Zxx) ),norm=colors.LogNorm())
            ax.set(xlabel="Time (s)",ylabel="Frequency (Hz)",title=f"{fname},{sname} Input 0")
            #fig.savefig(fr"{os.path.splitext(fn)[0]}-Input 0-spectrogram-{sname}{'-shifted' if shift else ''}.png")
            #plt.close(fig)

            f,t,Zxx = signal.spectrogram(i1,1e6)
            fig,ax = plt.subplots(constrained_layout=True)
            if shift:
                #plt.colorbar(ax.pcolormesh(t, np.fft.fftshift(f), np.fft.fftshift(Zxx, axes=0)**2),norm=colors.LogNorm())
                plt.colorbar(ax.pcolormesh(t, np.fft.fftshift(f), np.fft.fftshift(Zxx, axes=0)),norm=colors.LogNorm())
            else:
                #plt.colorbar(ax.pcolormesh(t, f, Zxx ** 2),norm=colors.LogNorm())
                plt.colorbar(ax.pcolormesh(t, f,  np.log10(Zxx) ),norm=colors.LogNorm())
            ax.set(xlabel="Time (s)",ylabel="Frequency (Hz)",title=f"{fname},{sname} Input 1")
            #fig.savefig(fr"{os.path.splitext(fn)[0]}-Input 1-spectrogram-{sname}{'-shifted' if shift else ''}.png")
            #plt.close(fig)
        return fig
    
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
    periods = list(filter(lambda x : fname in x,PERIODS))
    if periods:
        periods = periods[0][fname]
        # filter to time periods dict
        for sname,chunk in periods.items():
            # load sub bit of the stripe
            time,i0,i1 = loadTDMSSub(fn,chunk,is_time=True)
            #t=time
            #f,t,Zxx = plt.spectrogram(i0,1e6)
            Zxx, f, t, im = plt.specgram(i0, NFFT=8224, Fs=1e6) #8224
            fig1,ax = plt.subplots(constrained_layout=True)
            if shift:
                #plt.colorbar(ax.pcolormesh(t, np.fft.fftshift(f), np.fft.fftshift(Zxx, axes=0)**2),norm=colors.LogNorm())
                plt.colorbar(ax.pcolormesh(t, np.fft.fftshift(f), np.fft.fftshift(10*np.log10(Zxx), axes=0)), norm=colors.LogNorm())
            else:
                #plt.colorbar(ax.pcolormesh(t, f, Zxx ** 2),norm=colors.LogNorm())
                plt.colorbar(ax.pcolormesh(t, f, np.log10(Zxx) ),norm=colors.LogNorm())
            ax.set(xlabel="Time (s)",ylabel="Frequency (Hz)",title=f"{fname},{sname} Input 0")
           
            #f,t,Zxx = signal.spectrogram(i1,1e6)
            Zxx, f, t, im = plt.specgram(i1, NFFT=8224, Fs=1e6)
            fig2,ax = plt.subplots(constrained_layout=True)
            if shift:
                #plt.colorbar(ax.pcolormesh(t, np.fft.fftshift(f), np.fft.fftshift(Zxx, axes=0)**2),norm=colors.LogNorm())
                plt.colorbar(ax.pcolormesh(t, np.fft.fftshift(f), np.fft.fftshift(10*np.log10(Zxx), axes=0)),norm=colors.LogNorm())
            else:
                #plt.colorbar(ax.pcolormesh(t, f, Zxx ** 2),norm=colors.LogNorm())
                plt.colorbar(ax.pcolormesh(t, f, np.log10(Zxx) ),norm=colors.LogNorm())
            ax.set(xlabel="Time (s)",ylabel="Frequency (Hz)",title=f"{fname},{sname} Input 1")
            #fig.savefig(fr"{os.path.splitext(fn)[0]}-Input 1-spectrogram-{sname}{'-shifted' if shift else ''}.png")
            #plt.close(fig)
        return fig1, fig2


def WaveletTransformcwt(fn):#, wavelet, widths, dtype=None, **kwargs):
    '''Continuous wavelet transform.

    Performs a continuous wavelet transform on data, using the wavelet function. A CWT performs a convolution with data using the wavelet function, which is characterized by a width parameter and length parameter. The wavelet function is allowed to be complex.

    Parameters:
        data(N,) ndarray
        data on which to perform the transform.

    waveletfunction
    Wavelet function, which should take 2 arguments. The first argument is the number of points that the returned vector will have (len(wavelet(length,width)) == length). The second is a width parameter, defining the size of the wavelet (e.g. standard deviation of a gaussian). See ricker, which satisfies these requirements.

    widths(M,) sequence
    Widths to use for transform.

    dtypedata-type, optional
    The desired data type of output. Defaults to float64 if the output of wavelet is real and complex128 if it is complex.
    '''
    
    import numpy as np
    from scipy import signal
    import matplotlib.pyplot as plt
    
    #from matplotlib.colors import LogNorm
    fname = os.path.splitext(os.path.basename(fn))[0]
    # check if supported
    periods = list(filter(lambda x : fname in x,PERIODS))
    if periods:
        periods = periods[0][fname]
        # filter to time periods dict
        for sname,chunk in periods.items():
            # load sub bit of the stripe
            time,i0,i1 = loadTDMSSub(fn,chunk,is_time=True)
    
            widths = np.arange(1e5, 1e6, 1e5)
            cwtmatr = signal.cwt(i0, signal.ricker, widths)
            #fig1,ax = plt.subplots(constrained_layout=True)
            cwtmatr_yflip = np.flipud(cwtmatr)
            plt.imshow(cwtmatr_yflip, extent=[-1, 1, 0, 50000], cmap='PRGn', aspect='auto',
           vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
            plt.colorbar()
        plt.show()
            
def WaveletTransformPywtCwt(fn):#, wavelet, widths, dtype=None, **kwargs):        
    '''
    Continuous wavelet transform using the PyCWT scheme.
    This follows the form: coefs, freqs = pywt.cwt(signal, scales, wavelets)
        Inputs: signal - Data signal in array form
               scales - The scales to be used for cwt, in array form
               wavelets - The name of the wavelet used
                   Wavelet forms:
                       Mexican hat (mexh)
                       Morlet (morl)
                       Complex morlet (cmorB-C)
                       Gaussian derivative (gausP)
                       Complex gaussian derivative (cgauP)
                       Shannon (shanB-C)
                       Frequency B-spline (fbspB-C)
              Additional parameters:
                Method - Such as by convolution (conv) or fast Fourier transform (fft)
                Sampling period - 'The seconds taken for the output frequency'
        Output: coefs - The cwt coefficients
                freqs - Frequencies corresponding to the scale in array form
    
    '''
    import pywt
    import numpy as np
    import matplotlib.pyplot as plt    
        
    #from matplotlib.colors import LogNorm
    fname = os.path.splitext(os.path.basename(fn))[0]
    # check if supported
    periods = list(filter(lambda x : fname in x,PERIODS))
    if periods:
        periods = periods[0][fname]
        # filter to time periods dict
        for sname,chunk in periods.items():
            # load sub bit of the stripe
            time,i0,i1 = loadTDMSSub(fn,chunk,is_time=True)
            
            ##Zxx, f, t, im = plt.specgram(i0, NFFT=8224, Fs=1e6) #8224
            
            scales = np.arange(1, 31)  # No. of scales - Same as widths in above 
            #x=len(time)
            fig1,ax = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
            coef, freqs = pywt.cwt(i0, scales, 'gaus1', 1e6)  # Finding CWT using gaussian wavelet
            
            t = len(i0)
            print(t)
            #t2 = tuple((ti)/1e6 for ti in t)
            t2 = (t-1)/1e6            
            #t2 = tuple((ti)/1e6 for ti in t)
            #tms = (0,(t-1), 1e6)
            tms = (0, t2)
            
            #Plotting scalogram
            plt.figure(figsize=(15, 10))
            plt.imshow(abs(coef), extent=[tms, scales], interpolation='bilinear', cmap='bone',
           aspect='auto', vmax=abs(coef).max(), vmin=abs(coef).max())
            plt.gca().invert_yaxis()
            plt.yticks(np.arange(1, 31, 1))
            plt.xticks(np.arange(0, len(tms)+1, 10))
            ax.set(xlabel="Time or samples",ylabel="Frequency (Hz)",title=f"{fname},{sname} Input 0")
            plt.show()
            
            #for row in ax:
            #    for col in row:
            #        col.plot(x, y)
            #        col.plot(tms,i0)
# or use this method            
            #fig = plt.figure()
            #plt.subplot(1, 2, 1)
            #plt.plot(tms, i0)
            #plt.subplot(1, 2, 2)
            #ax.plot_surface(tms, freq, abs(coef))
            #plt.plot(x, y)
            #plt.show()


# NEEDS TO BE FINISHED AND FINE TUNED
            
def stackFilterStripes(path,freq=50e3,mode='highpass',**kwargs):
    '''
        Apply a highpass filter to each identified stripe in the file

        A Butterworth filter set to high-pass is applied to the target stripe.
        The original and filtered signal are plotted on the same axis and saved

        The order of the filter is set using order keyword.

        Generated plots are saved in the same location as the source file

        This is intended to remove the noise floor.

        Inputs:
            path : TDMS path
            freq : Cutoff freq. Default 50 KHz
            order : Filter order. Default 10.
    '''
    from scipy import signal
    sns.set_theme("paper")
    mode_dict = {'lp': "Low Pass",'hp':"High Pass",'bp':"Bandpass",'lowpass': "Low Pass",'highpass':"High Pass",'bandpass':"Bandpass"}
    # for each tdms
    for fn in glob(path):
        # check if supported
        periods = list(filter(lambda x : os.path.splitext(os.path.basename(fn))[0] in x,PERIODS))
        if periods:
            periods = periods[0][os.path.splitext(os.path.basename(fn))[0]]
            # make filter
            sos = signal.butter(kwargs.get("order",10), freq/(1e6/2), mode, fs=1e6, output='sos',analog=False)
            # plot period
            f_i0,ax_i0 = plt.subplots(constrained_layout=True)
            f_i1,ax_i1 = plt.subplots(constrained_layout=True)
            # filter to time periods dict
            for sname,chunk in periods.items():
                # load sub bit of the stripe
                time,i0,i1 = loadTDMSSub(path,chunk,is_time=True)
                # filter data
                filtered = signal.sosfilt(sos, i0)
                ax_i0.plot(time,filtered,label=sname)
                # filter data
                filtered = signal.sosfilt(sos, i1)
                ax_i1.plot(time,filtered,label=sname)

            ax_i0.legend(loc='upper left')
            ax_i0.set(xlabel="Time (s)",ylabel="Voltage (V)",title=f"{os.path.splitext(os.path.basename(fn))[0]}, Input 0, {sname}, {mode_dict[mode]}, {freq:.2E}Hz")
            f_i0.savefig(fr"{os.path.splitext(fn)[0]}-Input 0-time-stripe-overlap-filtered-{mode}-freq-{freq}.png")
            plt.close(f_i0)

            ax_i1.legend(loc='upper left')
            ax_i1.set(xlabel="Time (s)",ylabel="Voltage (V)",title=f"{os.path.splitext(os.path.basename(fn))[0]}, Input 1, {sname}, {mode_dict[mode]}, {freq:.2E}Hz")
            f_i1.savefig(fr"{os.path.splitext(fn)[0]}-Input 1-time-stripe-overlap-filtered-{mode}-freq-{freq}.png")
            plt.close(f_i1)
        else:
            print(f"Unsupported file {fn}!")

def periodogramStripes(path,**kwargs):
    ''' From scipy.signal.periodogram
    
    Inputs:
        
    path - TDMS file path
    
    Kwargs:
    fs - (float, optional) Sampling frequency of the x time series. Defaults to 1.0.
    window - (str or tuple or array_like, optional) Desired window to use. If window is a string or tuple, it is passed to get_window to generate the window values, which are DFT-even by default. See get_window for a list of windows and required parameters. If window is array_like it will be used directly as the window and its length must be equal to the length of the axis over which the periodogram is computed. Defaults to ‘boxcar’.
    nfftint - (optional) Length of the FFT used. If None the length of x will be used.
    detrend - (str or function or False, optional) Specifies how to detrend each segment. If detrend is a string, it is passed as the type argument to the detrend function. If it is a function, it takes a segment and returns a detrended segment. If detrend is False, no detrending is done. Defaults to ‘constant’.
    return_onesided - (bool, optional) If True, return a one-sided spectrum for real data. If False return a two-sided spectrum. Defaults to True, but for complex data, a two-sided spectrum is always returned.
    scaling - ({ ‘density’, ‘spectrum’ }, optional) Selects between computing the power spectral density (‘density’) where Pxx has units of V**2/Hz and computing the power spectrum (‘spectrum’) where Pxx has units of V**2, if x is measured in V and fs is measured in Hz. Defaults to ‘density’
    axis - (int, optional) Axis along which the periodogram is computed; the default is over the last axis (i.e. axis=-1).

Returns:

    f - (ndarray) Array of sample frequencies.
    Pxx - (ndarray) Power spectral density or power spectrum of x.



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
            # filter to time periods dict
            for sname,chunk in periods.items():
                # load sub bit of the stripe
                time,i0,i1 = loadTDMSSub(path,chunk,is_time=True)
                # plot period
                fig,ax = plt.subplots(constrained_layout=True)
                f, Pxx_den = signal.periodogram(i0, 1e6)
                ax.plot(f,Pxx_den)
                ax.set(xlabel="Frequency (Hz)",ylabel="PSD (V**2/Hz)",title=f"{ftitle}\nInput 0 {sname} Power Spectral Density")
                #fig.savefig(fr"{fname}-Input 0-{sname}-psd.png")
                #plt.close(fig)
        
                # plot period
                fig,ax = plt.subplots(constrained_layout=True)
                f, Pxx_den = signal.periodogram(i1, 1e6)
                ax.plot(f,Pxx_den)
                ax.set(xlabel="Frequency (Hz)",ylabel="PSD (V**2/Hz)",title=f"{ftitle}\nInput 1 {sname} Power Spectral Density")
                #fig.savefig(fr"{fname}-Input 1-{sname}-psd.png")
                #plt.close(fig)
                
                return fig
        else:
            print(f"Unsupported file {fn}!")

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
                fig,ax = plt.subplots(constrained_layout=True)
                f, Pxx_den = signal.welch(i0, 1e6,nperseg=8126)
                ax.plot(f,Pxx_den)
              #  ax.set(xlabel="Frequency (Hz)",ylabel="PSD (V**2/Hz)",title=f"{ftitle}\nInput 0 {sname} Power Spectral Density", fontsize=24)
                ax.set_xlabel("Frequency (Hz)", fontsize=14)
                ax.set_ylabel("PSD (V**2/Hz)", fontsize=14)
                ax.tick_params(axis='both', labelsize=14)
                if kwargs.get("freq_clip",None):
                    f0,f1 = kwargs.get("freq_clip",None)
                    ax.set_xlim(f0,f1)
                    Pmax = Pxx_den[(f>=f0)&(f<=f1)].max()
                    ax.set_ylim(0,Pmax+kwargs.get("yspace",0.10)*Pmax)
                  #  fig.savefig(fr"{fname}-Input 0-{sname}-welch-freq-clip-{f0}-{f1}.png")
               # else:
                  #  fig.savefig(fr"{fname}-Input 0-{sname}-welch.png")
                #plt.close(fig)
                return fig
        
                # plot period
                fig,ax = plt.subplots(constrained_layout=True)
                f, Pxx_den = signal.welch(i1, 1e6,nperseg=8126)
                ax.plot(f,Pxx_den)
                ax.set(xlabel="Frequency (Hz)",ylabel="PSD (V**2/Hz)",title=f"{ftitle}\nInput 1 {sname} Power Spectral Density")
                if kwargs.get("freq_clip",None):
                    f0,f1 = kwargs.get("freq_clip",None)
                    ax.set_xlim(f0,f1)
                    Pmax = Pxx_den[(f>=f0)&(f<=f1)].max()
                    ax.set_ylim(0,Pmax+kwargs.get("yspace",0.10)*Pmax)
                  #  fig.savefig(fr"{fname}-Input 1-{sname}-welch-freq-clip-{f0}-{f1}.png")
             #   else:
              #      fig.savefig(fr"{fname}-Input 1-{sname}-welch.png")
             #   plt.close(fig)
       # else:
         #   print(f"Unsupported file {fn}!")
                return fig

def welchStripesOverlap(path,**kwargs):
    '''
        Plot the Welch PSD of each stripe in the target file and OVERLAP all the plots on the same axis

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
    ftitle = os.path.splitext(os.path.basename(path))[0]
    # for each tdms
    for fn in glob(path):
        # check if supported
        periods = list(filter(lambda x : os.path.splitext(os.path.basename(fn))[0] in x,PERIODS))
        feedrate = list(filter(lambda x : os.path.splitext(os.path.basename(fn))[0] in x,FEED_RATE))
        if feedrate:
            feedrate = feedrate[0][ftitle] #Select Feedrate base on the source file
        if periods:
            periods = periods[0][ftitle]
            fig_i0,ax_i0 = plt.subplots(figsize=(15,9), constrained_layout=True)
            fig_i1,ax_i1 = plt.subplots(figsize=(15,9), constrained_layout=True)
            Pmax_i0 = 0
            Pmax_i1 = 0
            # filter to time periods dict
            for sname,chunk in periods.items():
                # load sub bit of the stripe
                time,i0,i1 = loadTDMSSub(path,chunk,is_time=True)
                # plot period
                f, Pxx_i0 = signal.welch(i0, 1e6,nperseg=4096) #1024 #8192
                if kwargs.get("freq_clip",None):
                    f0,f1 = kwargs.get("freq_clip",None)
                    Pmax_i0 = max(Pmax_i0,Pxx_i0[(f>=f0)&(f<=f1)].max())

                if kwargs.get("use_fr",False) and feedrate:
                    ax_i0.plot(f,Pxx_i0,label=feedrate[sname])
                else:
                    ax_i0.plot(f,Pxx_i0,label=sname)

                f, Pxx_i1 = signal.welch(i1, 1e6,nperseg=4096)
                if kwargs.get("freq_clip",None):
                    f0,f1 = kwargs.get("freq_clip",None)
                    Pmax_i1 = max(Pmax_i1,Pxx_i1[(f>=f0)&(f<=f1)].max())

                if kwargs.get("use_fr",False) and feedrate:
                    ax_i1.plot(f,Pxx_i1,label=feedrate[sname])
                else:
                    ax_i1.plot(f,Pxx_i1,label=sname)

            ax_i0.legend(fontsize=24)
            #ax_i0.set( xlabel="Frequency (Hz)",ylabel="PSD (V**2/Hz)",title=f"{ftitle}\nInput 0 Power Spectral Density")
            ax_i0.set_title("Input 0 Power Spectral Density")
            ax_i0.set_xlabel("Frequency (Hz)", fontsize=24)
            ax_i0.set_ylabel("PSD (V**2/Hz)",fontsize=24)
            ax_i0.tick_params(axis='both', labelsize=24)
            if kwargs.get("freq_clip",None):
                f0,f1 = kwargs.get("freq_clip",None)
                ax_i0.set_xlim(f0,f1)
                ax_i0.set_ylim(0,Pmax_i0+kwargs.get("yspace",0.10)*Pmax_i0)
                #fig_i0.savefig(fr"{fname}-Input 0-overlap-welch-freq-clip-{f0}-{f1}.png")
                
            else:
                fig_i0.savefig(fr"{fname}-Input 0-overlap-welch.png")

            ax_i1.legend(fontsize=24)
            #ax_i1.set(xlabel="Frequency (Hz)",ylabel="PSD (V**2/Hz)",title=f"{ftitle}\nInput 1 Power Spectral Density")
            ax_i1.set_title("Input 1 Power Spectral Density")
            ax_i1.set_xlabel("Frequency (Hz)",fontsize=24)
            ax_i1.set_ylabel("PSD (V**2/Hz)", fontsize=24)
            ax_i1.tick_params(axis='both', labelsize=24)
            if kwargs.get("freq_clip",None):
                f0,f1 = kwargs.get("freq_clip",None)
                ax_i1.set_xlim(f0,f1)
                ax_i1.set_ylim(0,Pmax_i1+kwargs.get("yspace",0.10)*Pmax_i1)
                #fig_i1.savefig(fr"{fname}-Input 1-overlap-welch-freq-clip-{f0}-{f1}.png")
                #return fig_i1
            else:
                fig_i1.savefig(fr"{fname}-Input 1-overlap-welch.png")
            return fig_i0, fig_i1
            plt.close('all')
        else:
            print(f"Unsupported file {fn}!")

def filterStripesProportion(path,minfreq=10e3,**kwargs):
    '''
        For each stripe, change the cut off frequency and record the ratio between the filtered max value and the unfiltered max value.

        The minfreq value is the minimum cutoff freq. Go too low and the gain will skyrocket due to close to 0 freqs being removed.

        The order of the filter is set using order keyword.

        Inputs:
            path : TDMS path
            minfreq : Minimum cutoff frequency. Default 10e3
            order : Filter order. Default 10.
            res : Resolution of the jumps. Default 1e3.
    '''
    from scipy import signal
    import multiprocessing as mp
    sns.set_theme("paper")
    # for each tdms
    for fn in glob(path):
        # check if supported
        periods = list(filter(lambda x : os.path.splitext(os.path.basename(fn))[0] in x,PERIODS))
        if periods:
            periods = periods[0][os.path.splitext(os.path.basename(fn))[0]]
            f,ax = plt.subplots(ncols=2,constrained_layout=True)
            # filter to time periods dict
            for sname,chunk in periods.items():
                # load sub bit of the stripe
                time,i0,i1 = loadTDMSSub(path,chunk,is_time=True)
                ## setup arrays for plotting
                # cutoff frequencies
                freq = [1e6/2,]
                # gain and set reference for Input 0
                gain_i0 = [1.0,]
                ref_i0 = i0.max()
                # gain and set reference for Input 1
                gain_i1 = [1.0,]
                ref_i1 = i1.max()
                for filt in np.arange(minfreq,1e6/2,kwargs.get("res",1e3))[::-1]:
                    # make filter
                    sos = signal.butter(kwargs.get("order",10), filt/(1e6/2), 'lowpass', fs=1e6, output='sos', analog=False)
                    # filter data
                    filtered = signal.sosfilt(sos, i0)
                    gain_i0.append(filtered.max()/ref_i0.max())
                    # filter data
                    filtered = signal.sosfilt(sos, i1)
                    gain_i1.append(filtered.max()/ref_i1.max())
                    # append cutoff freq
                    freq.append(filt)
                ax[0].plot(freq,gain_i0,label=sname)
                ax[1].plot(freq,gain_i1,label=sname)
            # setup legend
            ax[0].legend()
            ax[1].legend()
            # set labels
            ax[0].set(xlabel="Cuttoff Freq (Hz)",ylabel="Gain",title="Input 0")
            ax[1].set(xlabel="Cuttoff Freq (Hz)",ylabel="Gain",title="Input 1")
            f.suptitle(f"{os.path.splitext(os.path.basename(path))[0]}\nGain at different cuttoff freq")
            plt.close(f)
        else:
            print(f"Unsupported file {fn}!")

def filterStripesBP(path,lowcut=300e3,highcut=350e3,**kwargs):
    '''
        Apply a bandpass butterworth filter to each stripe in the target file and plot the result

        A Butterworth bandpass filter is applied to the signal.
        The original and filtered signals are plotted as separate traces on the same axis

        Generated plots are saved in the same location as the source file

        This is intended to investigate specific frequency bandsidentified in the STFT

        Inputs:
            path : TDMS file path
            lowcut : Low cut off frequency. Default 300 kHz
            highcut : High cut off frequency. Default 350 kHz
            order : Order of filter. Default 6.
    '''
    from scipy import signal
    for fn in glob(path):
        # check if supported
        periods = list(filter(lambda x : os.path.splitext(os.path.basename(fn))[0] in x,PERIODS))
        if periods:
            periods = periods[0][os.path.splitext(os.path.basename(fn))[0]]
            # filter to time periods dict
            for sname,chunk in periods.items():
                # load sub bit of the stripe
                time,i0,i1 = loadTDMSSub(path,chunk,is_time=True)
                # plot period
                f,ax = plt.subplots(constrained_layout=True)
                ax.plot(time,i0,'b-',label="Original")
                # make filter
                filtered = butter_bandpass_filter(i0, lowcut, highcut, order=kwargs.get("order",6))
                
                ax.plot(time,filtered,'r-',label="Filtered")
                ax.legend()
                ax.set(xlabel="Time (s)",ylabel="Voltage (V)",title="Input 0")
                # set title
                ax.set_title(f"{os.path.splitext(os.path.basename(fn))[0]}, Input 0, {sname}")
                ax.figure.savefig(fr"{os.path.splitext(fn)[0]}-Input 0-time-stripe-{sname}-filtered-freq-bp-{lowcut}-{highcut}.png")
                plt.close(f)

                # plot period
                f,ax = plt.subplots(constrained_layout=True)
                ax.plot(time,i1,'b-',label="Original")
                filtered = butter_bandpass_filter(i1, lowcut, highcut, order=kwargs.get("order",6))
                ax.plot(time,filtered,'r-',label="Filtered")
                ax.legend()
                ax.set(xlabel="Time (s)",ylabel="Voltage (V)",title="Input 1")
                # set title
                ax.set_title(f"{os.path.splitext(os.path.basename(fn))[0]}, Input 1, {sname}")
                ax.figure.savefig(fr"{os.path.splitext(fn)[0]}-Input 1-time-stripe-{sname}-filtered-freq-bp-{lowcut}-{highcut}.png")
                plt.close(f)
        else:
            print(f"Unsupported file {fn}!")

def stackFilterStripesBP(path,freq,**kwargs):
    '''
        Apply a bandpass butterworth filter to each stripe in the target file and plot the result
        ON THE SAME AXIS

        A Butterworth bandpass filter is applied to the signal.
        The original and filtered signals are plotted as separate traces on the same axis

        Generated plots are saved in the same location as the source file

        This is intended to investigate specific frequency bands identified in the STFT

        Inputs:
            path : TDMS file path
            lowcut : Low cut off frequency. Default 300 kHz
            highcut : High cut off frequency. Default 350 kHz
            order : Order of filter. Default 6.
    '''
    for fn in glob(path):
        # check if supported
        periods = list(filter(lambda x : os.path.splitext(os.path.basename(fn))[0] in x,PERIODS))
        if periods:
            periods = periods[0][os.path.splitext(os.path.basename(fn))[0]]
            # filter to time periods dict
            for sname,chunk in periods.items():
                print(sname)
                # load sub bit of the stripe
                time,i0,i1 = loadTDMSSub(path,chunk,is_time=True)
                # plot period
                f0,ax0 = plt.subplots(constrained_layout=True)
                ax0.plot(time,i0,label="Original")

                # plot period
                f1,ax1 = plt.subplots(constrained_layout=True)
                ax1.plot(time,i1,label="Original")
                # make filter
                for lowcut,highcut in freq:
                    print(lowcut,highcut)
                    filtered = butter_bandpass_filter(i0, lowcut, highcut, order=kwargs.get("order",6))
                    ax0.plot(time,filtered,label=f"{lowcut},{highcut} Hz")

                    filtered = butter_bandpass_filter(i1, lowcut, highcut, order=kwargs.get("order",6))
                    ax1.plot(time,filtered,label=f"{lowcut},{highcut} Hz")
                ax0.set_ylim(i0.min(),i0.max())
                ax0.legend()
                ax0.set(xlabel="Time (s)",ylabel="Voltage (V)",title=f"{os.path.splitext(os.path.basename(fn))[0]}, Input 0, {sname}")
                plt.show()
                #f0.savefig(fr"{os.path.splitext(fn)[0]}-Input 0-time-stripe-{sname}-filtered-freq-bp-stack.png")
                #plt.close(f0)

                ax1.set_ylim(i1.min(),i1.max())
                ax1.legend()
                ax1.set(xlabel="Time (s)",ylabel="Voltage (V)",title=f"{os.path.splitext(os.path.basename(fn))[0]}, Input 1, {sname}")
                
                #f1.savefig(fr"{os.path.splitext(fn)[0]}-Input 1-time-stripe-{sname}-filtered-freq-bp-stack.png")
                #plt.close(f1)
                return f0, f1
        else:
            print(f"Unsupported file {fn}!")

def stackFilterEdgesBP3D(path,freq,dist=int(50e3),**kwargs):
    from scipy import signal
    from scipy.signal import find_peaks
    import matplotlib.patches as mpatches
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from matplotlib.collections import PolyCollection

    def findEdgePoly3D(time,i0,freq):
        tt = []
        V = []
        ff = []
        # mask signal to +ve
        mask_filt = i0.copy()
        mask_filt[i0<0]=0
        # find peaks in the signal
        pks = find_peaks(mask_filt,distance=dist)[0]
        if len(pks)==0:
            raise ValueError(f"Failed to find +ve edge in data!")
        
        #pts.extend([[time[pk],mask_filt[pk],freq] for pk in pks])
        tt.extend(time[pks].tolist())
        V.extend(mask_filt[pks].tolist())
        ff.extend(len(pks)*[freq,])

        # mask signal to -ve
        mask_filt = i0.copy()
        mask_filt[i0>0]=0
        # find peaks in the signal
        pks = find_peaks(mask_filt,distance=dist)[0]
        if len(pks)==0:
            raise ValueError(f"Failed to find -ve edge in data!")
        pks = pks[::-1]
        #pts.extend([[time[pk],mask_filt[pk],freq] for pk in pks[::-1]])
        tt.extend(time[pks].tolist())
        V.extend(mask_filt[pks].tolist())
        ff.extend(len(pks)*[freq,])
        #return [list(zip(tt,ff,V))]
        print(len(tt),len(V),len(ff))
        return list(zip(tt,V))

    for fn in glob(path):
        # check if supported
        periods = list(filter(lambda x : os.path.splitext(os.path.basename(fn))[0] in x,PERIODS))
        if periods:
            periods = periods[0][os.path.splitext(os.path.basename(fn))[0]]
            # filter to time periods dict
            for sname,chunk in periods.items():
                print(sname)
                # load sub bit of the stripe
                time,i0,i1 = loadTDMSSub(path,chunk,is_time=True)
                # plot period
                #f0,ax0 = plt.subplots(constrained_layout=True)
                f0 = plt.figure()
                ax0 = f0.add_subplot(projection="3d")
                verts = findEdgePoly3D(time,i0,0)
                print(np.array(verts).shape)
                poly = PolyCollection([verts],facecolors=[plt.colormaps['viridis_r'](0.1),],alpha=.7)
                ax0.add_collection3d(poly,zs=0,zdir='y')
                plt.show()

                # plot period
                f1,ax1 = plt.subplots(constrained_layout=True)
                ax1.plot(time,i1,label="Original")
                # make filter
                for lowcut,highcut in freq:
                    print(lowcut,highcut)
                    filtered = butter_bandpass_filterplotFreqResponse(30e3,"lowpass")(i0, lowcut, highcut, order=kwargs.get("order",6))
                    ax0.plot(time,filtered,label=f"{lowcut},{highcut} Hz")

                    filtered = butter_bandpass_filter(i1, lowcut, highcut, order=kwargs.get("order",6))
                    ax1.plot(time,filtered,label=f"{lowcut},{highcut} Hz")
                ax0.set_ylim(i0.min(),i0.max())
                ax0.legend()
                ax0.set(xlabel="Time (s)",ylabel="Voltage (V)",title=f"{os.path.splitext(os.path.basename(fn))[0]}, Input 0, {sname}")
                plt.show()
                f0.savefig(fr"{os.path.splitext(fn)[0]}-Input 0-time-stripe-{sname}-filtered-freq-bp-stack.png")
                plt.close(f0)

                ax1.set_ylim(i1.min(),i1.max())
                ax1.legend()
                ax1.set(xlabel="Time (s)",ylabel="Voltage (V)",title=f"{os.path.splitext(os.path.basename(fn))[0]}, Input 1, {sname}")
                f1.savefig(fr"{os.path.splitext(fn)[0]}-Input 1-time-stripe-{sname}-filtered-freq-bp-stack.png")
                plt.close(f1)
        else:
            print(f"Unsupported file {fn}!")
            
#def WaveletAnalysis(path,freq,dist=int(50e3),**kwargs):
#    import numpy as np
#    import pywt
#    import matplotlib.pyplot as plt
#    
#    wavelet = 
    
    
if __name__ == "__main__":
    import matplotlib.colors as colors
    from scipy.signal import periodogram
    plt.rcParams['agg.path.chunksize'] = 10000
    #plotStripes("ae/sheff_lsbu_stripe_coating_3_pulsing.tdms")
    #drawEdgeAroundStripe("ae/sheff_lsbu_stripe_coating_3_pulsing.tdms",stripe_ref=0)

  ##  replotAE("ae/sheff_lsbu_stripe_coating_2.tdms",clip=True,ds=100)
    
#    plotLombScargle("ae/sheff_lsbu_stripe_coating_1.tdms",freqs=np.arange(300e3,350e3,1e3),normalize=True)
#    STFTWithLombScargle("ae/sheff_lsbu_stripe_coating_1.tdms",span=0.4,n_bins=100, grid_size=100,min_freq=300e3,max_freq=400e3)
    ## filterStripes
 #   filterStripes("ae/sheff_lsbu_stripe_coating_1.tdms",300e3,'highpass')
   # filterStripes("ae/sheff_lsbu_stripe_coating_2.tdms",1,'highpass')
 #   filterStripes("ae/sheff_lsbu_stripe_coating_3_pulsing.tdms",300e3,'highpass')
  #  plotSignalEnergyFreqBands("ae/sheff_lsbu_stripe_coating_2.tdms",10,tclip=None,fmin=0.1)
 
##    
##    stackFilterStripes("ae/sheff_lsbu_stripe_coating_1.tdms",300e3,'highpass')
##    stackFilterStripes("ae/sheff_lsbu_stripe_coating_2.tdms",300e3,'highpass')
##    stackFilterStripes("ae/sheff_lsbu_stripe_coating_3_pulsing.tdms",300e3,'highpass')

##    stackFilterStripesBP("ae/sheff_lsbu_stripe_coating_2.tdms",[[f0,f1] for f0,f1 in zip(np.arange(0.1,1e6/2,50e3),np.arange(0.1,1e6/2,50e3)[1:])])

##    filterStripesProportion("ae/sheff_lsbu_stripe_coating_2.tdms",minfreq=10e3)

    ## periodogramStripes
#    periodogramStripes("ae/sheff_lsbu_stripe_coating_2.tdms")
  ##  welchStripesOverlap("ae/sheff_lsbu_stripe_coating_2.tdms",freq_clip=[00e3,50e3],use_fr=True)
   # >> welchStripes("ae/sheff_lsbu_stripe_coating_2.tdms",freq_clip=[0,350e3],use_fr=True)
  ##  drawEdgeAroundStripe("ae/sheff_lsbu_stripe_coating_2.tdms",dist=int(50e3),mode="separate", stripe_ref = "stripe_1_4")
  
    plotSTFTStripes("ae/sheff_lsbu_stripe_coating_1.tdms",nperseg=256,fclip=None,use_log=True)
   # plotStripesLimits("ae/sheff_lsbu_stripe_coating_1.tdms")
    
##    welchStripesOverlap("ae/sheff_lsbu_stripe_coating_3_pulsing.tdms",freq_clip=[0,50e3],use_fr=False)
##    welchStripesOverlap("ae/sheff_lsbu_stripe_coating_2.tdms",freq_clip=[0,50e3],use_fr=False)
##    welchStripesOverlap("ae/sheff_lsbu_stripe_coating_1.tdms")
##    welchStripesOverlap("ae/sheff_lsbu_stripe_coating_2.tdms")
##    welchStripesOverlap("ae/sheff_lsbu_stripe_coating_3_pulsing.tdms")
##    filterStripesProportion("ae/sheff_lsbu_stripe_coating_2.tdms")

##    filterStripes("ae/sheff_lsbu_stripe_coating_1.tdms",50e3,mode="lowpass")
    #filterStripes("ae/sheff_lsbu_stripe_coating_2.tdms")
   # bandpassfilterStripes("ae/sheff_lsbu_stripe_coating_2.tdms",freq = [100e3, 250e3])
##    filterStripes("ae/sheff_lsbu_stripe_coating_3_pulsing.tdms",50e3,mode="lowpass")

    ## stripeSpectrogram
##    stripeSpectrogram("ae/sheff_lsbu_stripe_coating_1.tdms")
  #  stripeSpectrogram("ae/sheff_lsbu_stripe_coating_2.tdms")
    ##stripeSpecgram("ae/sheff_lsbu_stripe_coating_2.tdms")
    ##WaveletTransformcwt("ae/sheff_lsbu_stripe_coating_2.tdms")
    ##WaveletTransformPywtCwt("ae/sheff_lsbu_stripe_coating_2.tdms")
##    stripeSpectrogram("ae/sheff_lsbu_stripe_coating_3_pulsing.tdms")

 ##   plotStripesLimits("ae/sheff_lsbu_stripe_coating_1.tdms",cutoff_freq=[30e3,300e3],mode=["lowpass","highpass"],plot_mode="both")
##    plotStripesLimits("ae/sheff_lsbu_stripe_coating_2.tdms",cutoff_freq=[30e3,300e3],mode=["lowpass","highpass"],plot_mode="both")
##    plotStripesLimits("ae/sheff_lsbu_stripe_coating_3_pulsing.tdms",cutoff_freq=[30e3,300e3],mode=["lowpass","highpass"],plot_mode="both")
##
##    f = calcStripeAreas("ae/sheff_lsbu_stripe_coating_1.tdms")
##    f = calcStripeAreas("ae/sheff_lsbu_stripe_coating_1.tdms")
##    f = calcStripeAreas("ae/sheff_lsbu_stripe_coating_3_pulsing.tdms")
##    stackFilterEdgesBP3D("ae/sheff_lsbu_stripe_coating_1.tdms",[[f0,f1] for f0,f1 in zip(np.arange(0.1,1e6/2,50e3),np.arange(0.1,1e6/2,50e3)[1:])])
    #f.savefig("ae/sheff_lsbu_stripe_coating_2-signal-limits.png")
    #plt.close(f)
##    for stripe in STRIPE_PERIOD['sheff_lsbu_stripe_coating_1'].keys():
##        f = drawEdgeAroundStripe("ae/sheff_lsbu_stripe_coating_1.tdms",stripe_ref=stripe,mode="overlay")
##        f.savefig(f"ae/sheff_lsbu_stripe_coating_1-edge-trace-{stripe}-overlay.png")
##        plt.close(f)
    
##    f0,f1 = replotAE("ae/sheff_lsbu_stripe_coating_2.tdms",clip=False,ds=1)
##    f0.savefig("sheff_lsbu_stripe_coating_2-Input 0-time.png")
##    f1.savefig("sheff_lsbu_stripe_coating_2-Input 1-time.png")
##    plt.close('all')
        
        
