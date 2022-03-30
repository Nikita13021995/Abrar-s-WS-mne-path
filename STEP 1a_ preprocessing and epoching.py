# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 17:35:08 2022

@author: Nikita O
"""
# %% Used LIBRARIES upload

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mne 
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap)
from mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch
from mne.datasets import somato
#%matplotlib qt

############################################################################### BLOCK 1 - download
# %% Loading data for data-analysis

SDF      = 'G:\\WS' #Firstly we put in a DIRECT way a directory with the file
SDRF     = os.path.join(SDF, 'S28_Test1_tsss_mc_trans.fif') #Then we show the file in this directory
raw_data = mne.io.read_raw_fif(SDRF, allow_maxshield=False, preload=False, on_split_missing='raise', verbose=None) #Then we upload THE FILE from the directory

info     = raw_data.info
raw_data.plot()

# %% Selecting only gradiometers and STI channels

picks       = mne.pick_types(info, meg = 'grad', eog=True, ecg=True, stim=True, exclude=[]) #We can pick magnetometers with 'mag'

#saver of picked channels
raw_data.save('S28_Test1_tsss_mc_trans_grad_sti.fif', picks = picks, overwrite=True)

# %% Open new file with only certain channels
SDF         = 'G:\\WS' #Firstly we put in a DIRECT way a directory with the file
SDRF        = os.path.join(SDF, 'S28_Test1_tsss_mc_trans_grad_sti.fif') #Then we show the file in this directory
raw_data    = mne.io.read_raw_fif(SDRF, allow_maxshield=False, preload=False, on_split_missing='raise', verbose=None) #Then we upload THE FILE from the directory

# %% Loading data for data-analysis

SDF      = 'G:\\WS' #Firstly we put in a DIRECT way a directory with the file
SDRF     = os.path.join(SDF, 'S28_Test2_tsss_mc_trans.fif') #Then we show the file in this directory
raw_data_2 = mne.io.read_raw_fif(SDRF, allow_maxshield=False, preload=False, on_split_missing='raise', verbose=None) #Then we upload THE FILE from the directory

info_2     = raw_data_2.info

# %% Selecting only gradiometers and STI channels

picks_2       = mne.pick_types(info_2, meg = 'grad', eog=True, ecg=True, stim=True, exclude=[]) #We can pick magnetometers with 'mag'

#saver of picked channels
raw_data_2.save('S28_Test2_tsss_mc_trans_grad_sti.fif', picks = picks_2, overwrite=True)

# %% Open new file with only certain channels

SDF         = 'G:\\WS' #Firstly we put in a DIRECT way a directory with the file
SDRF        = os.path.join(SDF, 'S28_Test2_tsss_mc_trans_grad_sti.fif') #Then we show the file in this directory
raw_data_2    = mne.io.read_raw_fif(SDRF, allow_maxshield=False, preload=False, on_split_missing='raise', verbose=None) #Then we upload THE FILE from the directory

############################################################################### BLOCK 1 - merging 
# %% MERGING DATA FROM 2 FILES TO 1

#REC-STIM onset information
raw_data.first_samp
raw_data.plot()
raw_data.info['sfreq']
raw_data_time_of_first_sample = raw_data.first_samp / raw_data.info['sfreq'] #This is seconds difference between REC onset and STIM onset

raw_data_2.first_samp
raw_data_2.info['sfreq']
raw_data_time_of_first_sample_2 = raw_data_2.first_samp / raw_data_2.info['sfreq'] #This is seconds difference between REC onset and STIM onset


#Fot what we do alighning? 
events           = mne.find_events(raw_data, shortest_event=1, stim_channel='STI101') #can i do shortest event as 1?
events_2         = mne.find_events(raw_data_2, shortest_event=1, stim_channel='STI101') #can i do shortest event as 1?



#Block of alighning #1
events_3 = events.copy()
y = 0
while y < len(events_3[:,0]):
    events_3[y, 0] =  events_3[y, 0]  - raw_data.first_samp
    y += 1

raw_data.plot(start = -27, events=events_3) 
raw_data.info

#Block of alighning #2
events_4 = events_2.copy()
y = 0
while y < len(events_4[:,0]):
    events_4[y, 0] =  events_4[y, 0]  - raw_data_2.first_samp
    y += 1



raw_data_full     = raw_data.copy()
raw_data_full.append([raw_data_2])
raw_data_full.first_samp
events_full        = mne.find_events(raw_data_full, shortest_event=1, stim_channel='STI101') #can i do shortest event as 1?
raw_data_full.plot()
#Block of alighning #FULL 
events_f           = events_full.copy()
y = 0
while y            < len(events_f[:,0]):
    events_f[y, 0] =  events_f[y, 0]  - raw_data_full.first_samp
    y             += 1
    
# SAVER
np.save('S28_full-events.npy', events_f)
#events_full = np.load('S28_full-events.npy')

# %% EVENTS CHANGE
raw_data_full     = raw_data.copy()
raw_data_full.append([raw_data_2])
raw_data_full.first_samp
events_full        = mne.find_events(raw_data_full, shortest_event=1, stim_channel='STI101') #can i do shortest event as 1?

events = events_full.copy()

print(events[:,2])

a = 0
i = 0

while i < len(events[:,2]):     
    if events[i,2] == 155:
        a = i    
        if  events[a-1, 2] > 180 and events[a+1, 2] > 180:
            events[a,2] = 255
        else: 
            events[a,2] = 155
    i += 1 

print(events[:,2])
          
events_f           = events.copy()
y = 0

while y            < len(events_f[:,0]):
    events_f[y, 0] =  events_f[y, 0]  - raw_data_full.first_samp
    y             += 1
     
np.save('S28_full-events.npy', events_f)


# %% FILTERING DATA

#LOADING ANNOTATIONS AND CHECKING
annot_full = mne.read_annotations('S28_full-annotations.csv')
print(annot_full)
raw = raw_data_full
raw.set_annotations(annot_full)
raw.plot(events=events_full)
raw.info['lowpass']

raw_2            = raw.copy()
raw_2.load_data() #It is important to upload data to RAM



#NOTCH FILTER 
fig = raw.plot_psd(tmax=np.inf, fmax=250, average=True)
#pick = raw_2.pick_types(meg='grad')
list = ['MEG0113',  'MEG0112',  'MEG0122',  'MEG0123',  'MEG0132',  'MEG0133',  'MEG0143',
 'MEG0142',  'MEG0213',  'MEG0212',  'MEG0222',  'MEG0223',  'MEG0232',  'MEG0233',  'MEG0243',
 'MEG0242',  'MEG0313',  'MEG0312',  'MEG0322',  'MEG0323',  'MEG0333',  'MEG0332',  'MEG0343',
 'MEG0342', 'MEG0413', 'MEG0412', 'MEG0422', 'MEG0423', 'MEG0432', 'MEG0433', 'MEG0443',
 'MEG0442', 'MEG0513', 'MEG0512', 'MEG0523', 'MEG0522', 'MEG0532', 'MEG0533', 'MEG0542',
 'MEG0543', 'MEG0613', 'MEG0612', 'MEG0622', 'MEG0623', 'MEG0633', 'MEG0632', 'MEG0642',
 'MEG0643', 'MEG0713', 'MEG0712', 'MEG0723', 'MEG0722', 'MEG0733', 'MEG0732', 'MEG0743',
 'MEG0742', 'MEG0813', 'MEG0812', 'MEG0822', 'MEG0823', 'MEG0913', 'MEG0912', 'MEG0923',
 'MEG0922', 'MEG0932', 'MEG0933', 'MEG0942', 'MEG0943', 'MEG1013', 'MEG1012', 'MEG1023',
 'MEG1022', 'MEG1032', 'MEG1033', 'MEG1043', 'MEG1042', 'MEG1112', 'MEG1113', 'MEG1123',
 'MEG1122', 'MEG1133', 'MEG1132', 'MEG1142', 'MEG1143', 'MEG1213', 'MEG1212', 'MEG1223',
 'MEG1222', 'MEG1232', 'MEG1233', 'MEG1243', 'MEG1242', 'MEG1312', 'MEG1313', 'MEG1323',
 'MEG1322', 'MEG1333', 'MEG1332', 'MEG1342', 'MEG1343', 'MEG1412', 'MEG1413', 'MEG1423',
 'MEG1422', 'MEG1433', 'MEG1432', 'MEG1442', 'MEG1443', 'MEG1512', 'MEG1513',
 'MEG1522', 'MEG1523', 'MEG1533', 'MEG1532', 'MEG1543', 'MEG1542', 'MEG1613', 'MEG1612',
 'MEG1622', 'MEG1623', 'MEG1632', 'MEG1633', 'MEG1643', 'MEG1642', 'MEG1713', 'MEG1712',
 'MEG1722', 'MEG1723', 'MEG1732', 'MEG1733', 'MEG1743', 'MEG1742', 'MEG1813', 'MEG1812',
 'MEG1822', 'MEG1823', 'MEG1832', 'MEG1833', 'MEG1843', 'MEG1842', 'MEG1912', 'MEG1913',
 'MEG1923', 'MEG1922', 'MEG1932', 'MEG1933', 'MEG1943', 'MEG1942', 'MEG2013', 'MEG2012',
 'MEG2023', 'MEG2022', 'MEG2032', 'MEG2033', 'MEG2042', 'MEG2043', 'MEG2113', 'MEG2112',
 'MEG2122', 'MEG2123', 'MEG2133', 'MEG2132', 'MEG2143', 'MEG2142', 'MEG2212',
 'MEG2213', 'MEG2223', 'MEG2222', 'MEG2233', 'MEG2232', 'MEG2242', 'MEG2243', 'MEG2312',
 'MEG2313', 'MEG2323', 'MEG2322', 'MEG2332', 'MEG2333', 'MEG2343', 'MEG2342', 'MEG2412',
 'MEG2413', 'MEG2423', 'MEG2422', 'MEG2433', 'MEG2432', 'MEG2442', 'MEG2443', 'MEG2512',
 'MEG2513', 'MEG2522', 'MEG2523', 'MEG2533', 'MEG2532', 'MEG2543', 'MEG2542', 'MEG2612',
 'MEG2613', 'MEG2623', 'MEG2622', 'MEG2633', 'MEG2632', 'MEG2642', 'MEG2643']


def add_arrows(axes):
    # add some arrows at 60 Hz and its harmonics
    for ax in axes:
        freqs = ax.lines[-1].get_xdata()
        psds = ax.lines[-1].get_ydata()
        for freq in (50, 100, 150, 200):
            idx = np.searchsorted(freqs, freq)
            # get ymax of a small region around the freq. of interest
            y = psds[(idx - 4):(idx + 5)].max()
            ax.arrow(x=freqs[idx], y=y + 18, dx=0, dy=-12, color='red',
                     width=0.1, head_width=3, length_includes_head=True

freq = [50]

raw_2_notch_fit = raw_2.copy().notch_filter(
    freqs=freq, picks=list, method='spectrum_fit', filter_length='10s')
for title, data in zip(['Un', 'spectrum_fit '], [raw, raw_notch_fit]):
    fig = data.plot_psd(fmax=250, average=True)
    fig.subplots_adjust(top=0.85)
    fig.suptitle('{}filtered'.format(title), size='xx-large', weight='bold')
    add_arrows(fig.axes[:2])
    
fig = raw_2_notch_fit.plot_psd(fmax=250, average=True)
add_arrows(fig.axes[:2])






#APPLYING ICA - MORE FOR INFO
raw_3 = raw_2_notch_fit

ecg_evoked       = create_ecg_epochs(raw_3).average()
ecg_evoked.apply_baseline() 
ecg_evoked.plot_joint()
eog_evoked       = create_eog_epochs(raw_3).average()
eog_evoked.apply_baseline()
eog_evoked.plot_joint()




#REAL ICA
filt_raw         = raw_3.copy().filter(l_freq=1, h_freq=80) #As bandpass we have a range from 1 to 80 #Abrar used this frequency

# set up and fit the ICA
ica              = mne.preprocessing.ICA(n_components=40, random_state=97, max_iter=800)
ica_filt_raw = ica.fit(filt_raw)  #reject_by_annotation - default TRUE, reject - can give reject dict here

ica.plot_sources(filt_raw, show_scrollbars=False)
ica.plot_components(sensors = True, colorbar = True, outlines = 'skirt')
###ica.plot_properties(filt_raw, picks=19)

raw_4 = filt_raw  
raw_4.plot()




#ADDITIONAL CHECK WITH EOG AND ECG CHANNELS

ica.exclude = []
#Find which ICs match the EOG pattern
eog_indices, eog_scores = ica.find_bads_eog(raw_4)
ica.exclude = eog_indices

ica.plot_scores(eog_scores, exclude=eog_indices)
ica.plot_properties(raw_4, picks=eog_indices)
ica.plot_sources(eog_evoked)

#Find which ICs capture ECG pattern
ecg_indices, ecg_scores = ica.find_bads_ecg(raw_4, method='correlation', threshold='auto')

ica.plot_scores(ecg_scores, exclude=ecg_indices)
ica.plot_properties(raw_4, picks=ecg_indices)
ica.plot_sources(ecg_evoked)

#MERGING ECG and EOG
ica.plot_sources(raw_4, show_scrollbars=False)
ica.exclude= ica.exclude + ecg_indices

ica.exclude
ica.save('S28_full-ica.fif')
##mne.preprocessing.read_ica('S28_full-ica.fif', verbose=None)


#FINAL DECISION ABOUT DATA RECONSTRUCTION
raw_5 = raw_4.copy()
ica.apply(raw_5)
raw_4.plot()
raw_5.plot()

raw.save('S28_full_reconst.fif', overwrite=True)


#mne.preprocessing.read_ica
# %% EPOCHING and ANNOTATING

events           = mne.find_events(raw_5, shortest_event=1, stim_channel='STI101') #can i do shortest event as 1?
events_total = events.copy()

#I already do this, but perhaps it is better to do it here INSIDE
y = 0
while y < len(events_total[:,0]):
    events_total[y, 0] =  events_total[y, 0]  - raw_5.first_samp
    y += 1
a = 0
i = 0
while i < len(events_total[:,2]):     
    if events_total[i,2] == 155:
        a = i    
        if  events_total[a-1, 2] > 180 and events_total[a+1, 2] > 180:
            events_total[a,2] = 255
        else: 
            events_total[a,2] = 155
    i += 1 




raw_5.plot(events=events) #Here I use not EVENTS_TOTAL, because the function already aliogn the data

event_dict = {'Instruction/Spatial': 100, ' Instruction/Temporal': 200, 
              'Stimulus1/Spatial': 101, 
              'Stimulus2/Spatial': 102, 'Stimulus3/Spatial': 103, 'Stimulus4/Spatial': 104, 'Delay/Spatial': 155, 'Probe/Spatial': 105, 
              'CorResp/Spatial': 110, 'InCorResp/Spatial': 120, 
              'Stimulus1/Temporal': 201,
              'Stimulus2/Temporal': 202, 'Stimulus3/Temporal': 203, 'Stimulus4/Temporal': 204, 'Probe/Temporal': 205, 
              'CorResp/Temporal': 210, 'InCorResp/Temporal': 220, 'Delay/Temporal': 255}


raw_5.plot(events=events, color='gray',
         event_color={100: 'r', 200: 'g', 101: 'b', 102: 'b', 103: 'b', 104: 'b',
                      155: 'k', 110: 'm', 120: 'm', 
                      201: 'b', 202: 'b', 203: 'b', 204: 'b',
                      210: 'm', 220: 'm', 255: 'k'})

#REJECT CRITERIA
reject_criteria = dict(grad=3000e-13)    # 3000 fT/cm
flat_criteria = dict(grad=1e-13)         # 1 fT/cm

raw_5.plot()
#EPOCHING!

epochs = mne.Epochs(raw_5, events_total, event_id=event_dict, tmin=-8, tmax=8, preload=True) # FULL
epochs = mne.Epochs(raw_5, events_total, event_id=event_dict, tmin=-8, tmax=8, preload=True) # FULL

epochs_s = mne.Epochs(raw_5, events, event_id=155, tmin=-8, tmax=8, reject = reject_criteria, flat=flat_criteria, preload=True)
epochs_r = mne.Epochs(raw_5, events, event_id=101, reject = reject_criteria, flat=flat_criteria, preload=True)

epochs_t = mne.Epochs(raw_5, events, event_id=255, tmin=-8, tmax=8, reject = reject_criteria, flat=flat_criteria, preload=True)

epochs.save('S28_t1_epochs-epo.fif', overwrite=True)
epochs_s.save('S28_t1_S_epochs-epo.fif', overwrite=True)
epochs_t.save('S28_t1_T_epochs-epo.fif', overwrite=True)
#epochs_from_file = mne.read_epochs('saved-audiovisual-epo.fif', preload=False)

frequencies =    np.arange(7, 30, 3)
power = mne.time_frequency.tfr_morlet(epochs_s, n_cycles=2, return_itc=False,
                                      freqs=frequencies, decim=3)
power.plot()

del ecg_scores, eog_evoked, events_2, events_3, events_4, events_f, events_full, fig, filt_raw, ica, ica_filt_raw, raw, raw_2, raw_2_notch_fit, raw_3, raw_4, raw_data, raw_data_2, raw_data_full #MEMORY SAVER





