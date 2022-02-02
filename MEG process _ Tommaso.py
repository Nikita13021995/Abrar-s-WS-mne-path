# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 10:32:46 2022

@author: Nikita O
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 25 20:39:59 2022

@author: Nikita O
"""

# %% Used LIBRARIES upload

import os
import numpy as np
import matplotlib.pyplot as plt
import mne 
from mne.preprocessing import (ICA, create_eog_epochs, create_ecg_epochs,
                               corrmap)
from mne.time_frequency import tfr_morlet, psd_multitaper, psd_welch
from mne.datasets import somato
#%matplotlib qt


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

# %% MERGING DATA FROM 2 FILES TO 1

raw_data.plot() 
raw_data_2.plot() 
raw_data_full = raw_data.copy()

raw_data_full.append([raw_data_2])
raw_data_full.plot() 

# %% FILTERING DATA

raw = raw_data
raw.plot()

#Here I have it just to look at which moment there were ocular artifacts and did they happen during particular event
events           = mne.find_events(raw_data, shortest_event=1, stim_channel='STI101') #can i do shortest event as 1?
raw_data.plot(events=events)

### QUESTION - do we have a biased onset of sampling counting in NEUROMAG??? 
#print(raw_data.info['sfreq'], 'Hz')            
#time_of_first_sample = raw_data.first_samp / raw_data.info['sfreq'] #Time of the first sample to happen


raw_2            = raw.copy()
raw_2.load_data() #It is important to upload data to RAM

#APPLYING ICA

ecg_evoked       = create_ecg_epochs(raw_2).average()
ecg_evoked.apply_baseline() #I don't know any baseline here
ecg_evoked.plot_joint()

eog_evoked       = create_eog_epochs(raw_2).average()
eog_evoked.apply_baseline()
eog_evoked.plot_joint()

filt_raw         = raw_2.copy().filter(l_freq=1., h_freq=80) #As bandpass we have a range from 1 to 80 #Abrar used this frequency

# set up and fit the ICA
ica              = mne.preprocessing.ICA(n_components=40, random_state=97, max_iter=800)
ica_filt_raw = ica.fit(filt_raw)  

ica.plot_sources(raw_2, show_scrollbars=False)
ica.plot_components(sensors = True, colorbar = True, outlines = 'skirt')
ica.plot_properties(raw_2)

#ADDITIONAL CHECK WITH EOG AND ECG CHANNELS

ica.exclude = []
# find which ICs match the EOG pattern
eog_indices, eog_scores = ica.find_bads_eog(raw_2)
ica.exclude = eog_indices
# barplot of ICA component "EOG match" scores
ica.plot_scores(eog_scores)
# plot diagnostics
ica.plot_properties(raw_2, picks=eog_indices)
# plot ICs applied to raw data, with EOG matches highlighted
ica.plot_sources(raw_2, show_scrollbars=False)
# plot ICs applied to the averaged EOG epochs, with EOG matches highlighted
ica.plot_sources(eog_evoked)

ica.exclude = []
# find which ICs match the ECG pattern
ecg_indices, ecg_scores = ica.find_bads_ecg(raw_2, method='correlation', threshold='auto')
ica.exclude = ecg_indices
# barplot of ICA component "ECG match" scores
ica.plot_scores(ecg_scores)
# plot diagnostics
ica.plot_properties(raw_2, picks=ecg_indices)
# plot ICs applied to raw data, with ECG matches highlighted
ica.plot_sources(raw_2, show_scrollbars=False)
# plot ICs applied to the averaged ECG epochs, with ECG matches highlighted
ica.plot_sources(ecg_evoked)

#FINAL DECISION ABOUT DATA RECONSTRUCTION
ica.exclude = [0, 25, 39]
reconst_raw = raw_2.copy()
ica.apply(reconst_raw)
raw_2.plot()
reconst_raw.plot()

# %% EPOCHING and ANNOTATING

events           = mne.find_events(reconst_raw, shortest_event=1, stim_channel='STI101') #can i do shortest event as 1?
reconst_raw.plot(events=events)

event_dict = {'Instruction/Spatial': 100, ' Instruction/Temporal': 200, 
              'Stimulus1/Spatial': 101, 
              'Stimulus2/Spatial': 102, 'Stimulus3/Spatial': 103, 'Stimulus4/Spatial': 104, 'Delay': 155, 'Probe/Spatial': 105, 
              'CorResp/Spatial': 110, 'InCorResp/Spatial': 120, 
              'Stimulus1/Temporal': 201,
              'Stimulus2/Temporal': 202, 'Stimulus3/Temporal': 203, 'Stimulus4/Temporal': 204, 'Probe/Temporal': 205, 
              'CorResp/Temporal': 210, 'InCorResp/Temporal': 220}


reconst_raw.plot(events=events, color='gray',
         event_color={100: 'r', 200: 'g', 101: 'b', 102: 'b', 103: 'b', 104: 'b',
                      155: 'k', 110: 'm', 120: 'm', 
                      201: 'b', 202: 'b', 203: 'b', 204: 'b',
                      210: 'm', 220: 'm'})

#ANY ANNOTATIONS TO THE FILE 
# my_annot = mne.Annotations(onset=[15, 31, 48],  # in seconds
#                            duration=[16, 16, 16],  # in seconds, too
#                            description=['Trial 1', 'Trial 2', 'Trial 3'])
# rawRec = reconst_raw.copy()
# rawRec.plot()
# rawRec.set_annotations()
# rawRec.set_annotations(my_annot)
# rawRec.plot(events=events)
# rawRec.annotations.save('S28_t1_annotations.csv', overwrite=True)
# annot_from_file = mne.read_annotations('S28_t1_annotations.csv')
# print(annot_from_file)

#EPOCHING!
epochs = mne.Epochs(reconst_raw, events, event_id=event_dict, tmin=-8, tmax=8, preload=True)
epochs.save('S28_t1_epochs-epo.fif', overwrite=True)
#del epochs #MEMORY SAVER
#epochs.plot()

spat_epochs = epochs['Spatial']
temp_epochs = epochs['Temporal']
del epochs  # free up memory


spat_epochs.plot_image()
temp_epochs.plot_image()

#### PSD part
#### Operations are too heavy to conduct them =(
spat_epochs.plot_psd(fmin=1., fmax=80., average=True, spatial_colors=False)
temp_epochs.plot_psd(fmin=1., fmax=80., average=True, spatial_colors=False)

spat_evoked = spat_epochs.average()
temp_evoked = temp_epochs.average()

mne.viz.plot_compare_evokeds(dict(spation=spat_evoked, temporal=temp_evoked),
                              legend='upper left', show_sensors='upper right')


spat_evoked.plot_joint()
spat_evoked.plot_topomap(times=[0., 0.08, 0.1, 0.12, 0.2])

evoked_diff = mne.combine_evoked([spat_evoked, temp_evoked], weights=[1, -1])
evoked_diff.pick_types(meg='grad').plot_topo(color='r', legend=False)

