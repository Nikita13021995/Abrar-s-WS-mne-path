# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 17:31:31 2022

@author: Nikita O
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 10:10:11 2022

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

# %% Open new file with only certain channels
index = 3

SDF                = 'G:\\WS\FULL DATA' #Firstly we put in a DIRECT way a directory with the file
SDRF               = os.path.join(SDF, 'S{}_full_reconst.fif'.format(index)) #Then we show the file in this directory
raw_data           = mne.io.read_raw_fif(SDRF, allow_maxshield=False, preload=False, on_split_missing='raise', verbose=None) #Then we upload THE FILE from the directory
SDRF               = os.path.join(SDF, 'S{}_Test2_tsss_mc_trans_grad_sti.fif'.format(index)) #Then we show the file in this directory
raw_data_2         = mne.io.read_raw_fif(SDRF, allow_maxshield=False, preload=False, on_split_missing='raise', verbose=None) #Then we upload THE FILE from the directory


# %% MERGING DATA FROM 2 FILES TO 1

raw_data_full      = raw_data.copy()
raw_data_full.append([raw_data_2])
events_full        = mne.find_events(raw_data_full, shortest_event=1, stim_channel='STI101') #can i do shortest event as 1?
events = events_full.copy()
raw_data_full.plot()

raw_data.find_events()
events_full        = mne.find_events(raw_data, shortest_event=1, stim_channel='STI101') #can i do shortest event as 1?

# # SAVER
# np.save('S20_full-events.npy', events_full)
# #events_full = np.load('S28_full-events.npy')


# %% NOTCH FILTER
raw = raw_data_full.copy()
raw.load_data() #It is important to upload data to RAM

fig = raw.plot_psd(tmax=np.inf, fmax=80, average=True)

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

filt_raw         = raw.copy().filter(l_freq=1, h_freq=80) #As bandpass we have a range from 1 to 80 #Abrar used this frequency

freq = [50]
raw_notch_fit = filt_raw.notch_filter(
    freqs=freq, picks=list, method='spectrum_fit', filter_length='10s')

fig = raw_notch_fit.plot_psd(fmax=80, average=True)
raw_notch_fit.plot()



# %% ANNOTATIONS
raw_2 = raw_notch_fit.copy()
raw_2.plot(events=events)
raw_2.annotations.save('S{}_full-annotations.csv'.format(index), overwrite = False)

# %% FILTERING DATA

raw = raw_2.copy()

#LOADING ANNOTATIONS AND CHECKING

annot_full = mne.read_annotations('S{}_full-annotations.csv'.format(index))
print(annot_full)
raw.set_annotations(annot_full)
raw.plot(events=events_full)


raw_3            = raw.copy()
raw_3.load_data() #It is important to upload data to RAM


# set up and fit the ICA
ica              = mne.preprocessing.ICA(n_components=40, random_state=97, max_iter=800)
ica_filt_raw = ica.fit(raw_3)  #reject_by_annotation - default TRUE, reject - can give reject dict here

ica.plot_sources(raw_3, show_scrollbars=False)
ica.plot_components(sensors = True, colorbar = True, outlines = 'skirt')






#ADDITIONAL CHECK WITH EOG AND ECG CHANNELS
raw_4 = raw_3

ica.exclude = []
#Find which ICs match the EOG pattern
eog_indices, eog_scores = ica.find_bads_eog(raw_4)
ica.exclude = eog_indices

# ica.plot_scores(eog_scores, exclude=eog_indices)
# ica.plot_properties(raw_4, picks=eog_indices)
# ica.plot_sources(eog_evoked)

#Find which ICs capture ECG pattern
ecg_indices, ecg_scores = ica.find_bads_ecg(raw_4, method='correlation', threshold='auto')

# ica.plot_scores(ecg_scores, exclude=ecg_indices)
# ica.plot_properties(raw_4, picks=ecg_indices)
# ica.plot_sources(ecg_evoked)

#MERGING ECG and EOG

ica.plot_sources(raw_4, show_scrollbars=False)
ica.exclude= ica.exclude + ecg_indices

ica.exclude    = [0, 9,23, 32]
ica.save('S{}_full-ica.fif'.format(index))

ica = mne.preprocessing.read_ica('S12_full-ica.fif', verbose=None)


#FINAL DECISION ABOUT DATA RECONSTRUCTION
raw_5 = raw_4.copy()
ica.apply(raw_5)
raw_4.plot()
raw_5.plot()

raw_5.save('S{}_full_reconst.fif'.format(index), overwrite=False)


#mne.preprocessing.read_ica


# %% EPOCHING and ANNOTATING

del a, ecg_evoked, eog_evoked, fig, filt_raw, freq, i, ica, raw, raw_2, raw_3, raw_4, raw_data, raw_data_2, raw_data_full, raw_notch_fit, y

SDF                = 'G:\\WS\FULL DATA' #Firstly we put in a DIRECT way a directory with the file
SDRF               = os.path.join(SDF, 'S16_full_reconst.fif') #Then we show the file in this directory
raw_5 = mne.io.read_raw_fif(SDRF, allow_maxshield=False, preload=False, on_split_missing='raise', verbose=None) #Then we upload THE FILE from the directory


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


raw_5.plot(events=events) #Here I use not EVENTS_TOTAL, because the function already aliogn the data

event_dict = {'Instruction/Spatial': 100, ' Instruction/Temporal': 200, 
              'Stimulus1/Spatial': 101, 
              'Stimulus2/Spatial': 102, 'Stimulus3/Spatial': 103, 'Stimulus4/Spatial': 104, 'Delay/Spatial': 155, 'Probe/Spatial': 105, 
              'CorResp/Spatial': 110, 'InCorResp/Spatial': 120, 
              'Stimulus1/Temporal': 201,
              'Stimulus2/Temporal': 202, 'Stimulus3/Temporal': 203, 'Stimulus4/Temporal': 204, 'Probe/Temporal': 205, 
              'CorResp/Temporal': 210, 'InCorResp/Temporal': 220, 'Delay/Temporal': 255}


# raw_5.plot(events=events, color='gray',
#          event_color={100: 'r', 200: 'g', 101: 'b', 102: 'b', 103: 'b', 104: 'b',
#                       155: 'k', 110: 'm', 120: 'm', 
#                       201: 'b', 202: 'b', 203: 'b', 204: 'b',
#                       210: 'm', 220: 'm', 255: 'k'})

#REJECT CRITERIA
reject_criteria = dict(grad=3000e-13)    # 3000 fT/cm
flat_criteria = dict(grad=1e-13)         # 1 fT/cm





#EPOCHING!
epochs = mne.Epochs(raw_5, events_total, event_id=event_dict, tmin=-8, tmax=8, preload=True) # FULL
epochs_s = mne.Epochs(raw_5, events, event_id=155, tmin=-8, tmax=8, reject = reject_criteria, flat=flat_criteria, preload=True)
epochs_t = mne.Epochs(raw_5, events, event_id=255, tmin=-8, tmax=8, reject = reject_criteria, flat=flat_criteria, preload=True)
epochs_sa = mne.Epochs(raw_5, events, event_id=155, tmin=0, tmax=4, reject = reject_criteria, flat=flat_criteria, preload=True, baseline=(0,0))
epochs_ta = mne.Epochs(raw_5, events, event_id=255, tmin=0, tmax=4, reject = reject_criteria, flat=flat_criteria, preload=True, baseline=(0,0))


epochs.save('S{}_full_epochs-epo.fif'.format(index), overwrite=True)
epochs_s.save('S{}_full_S_epochs-epo.fif'.format(index), overwrite=False)
epochs_t.save('S{}_full_T_epochs-epo.fif'.format(index), overwrite=False)
epochs_sa.save('S{}_full_Sa_epochs-epo.fif'.format(index), overwrite=False)
epochs_ta.save('S{}_full_Ta_epochs-epo.fif'.format(index), overwrite=False)

del a, annot_full, ecg_indices, ecg_scores, eog_indices, eog_scores, events_full, events_f, i, ica_filt_raw

del epochs_s, epochs_t





