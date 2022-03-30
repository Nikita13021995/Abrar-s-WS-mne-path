# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 11:06:49 2022

@author: Nikita O
"""
# %% LIBRARIES
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

from mne.cov import compute_covariance
from mne.datasets import somato
from mne.time_frequency import csd_morlet
from mne.beamformer import (make_dics, apply_dics_csd, make_lcmv,
                            apply_lcmv_cov)
from mne.minimum_norm import (make_inverse_operator, apply_inverse_cov)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from fooof import FOOOF
from fooof.sim.gen import gen_aperiodic
from fooof.plts.spectra import plot_spectrum, plot_spectra
from fooof.plts.annotate import plot_annotated_peak_search

from mne.stats import permutation_cluster_test
from mne.stats import spatio_temporal_cluster_test

#CHEKING THE VERSION OF MNE
mne.__version__

# %% LOADING

######### FOR LOOP LOADER
i = 3
power_ls = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
power_lt = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
len(power_lt)
len(power_ls)

for i in range(3,29): 
    power_s = mne.time_frequency.read_tfrs('S{}_power_s-tfr.h5'.format(i))
    power_ls[i-4] = power_s[0]  
    i+=1

i = 3
for i in range(3,29): 
    power_t = mne.time_frequency.read_tfrs('S{}_power_t-tfr.h5'.format(i))
    power_lt[i-4] = power_t[0]  
    i+=1
    
epochs_s_28 = mne.read_epochs('S28_full_S_epochs-epo.fif', preload=False)

######### MANUAL LOADER
power_s_28 = mne.time_frequency.read_tfrs('S28_power_s-tfr.h5')
power_s_28_1 = power_s_28[0]
power_s_27 = mne.time_frequency.read_tfrs('S27_power_s-tfr.h5')
power_s_27_1 = power_s_27[0]
power_s_26 = mne.time_frequency.read_tfrs('S26_power_s-tfr.h5')
power_s_26_1 = power_s_26[0]
power_s_25 = mne.time_frequency.read_tfrs('S25_power_s-tfr.h5')
power_s_25_1 = power_s_25[0]
power_s_24 = mne.time_frequency.read_tfrs('S24_power_s-tfr.h5')
power_s_24_1 = power_s_24[0]
power_s_23 = mne.time_frequency.read_tfrs('S23_power_s-tfr.h5')
power_s_23_1 = power_s_23[0]
power_s_22 = mne.time_frequency.read_tfrs('S22_power_s-tfr.h5')
power_s_22_1 = power_s_22[0]
power_s_21 = mne.time_frequency.read_tfrs('S21_power_s-tfr.h5')
power_s_21_1 = power_s_21[0]
power_s_20 = mne.time_frequency.read_tfrs('S20_power_s-tfr.h5')
power_s_20_1 = power_s_20[0]
power_s_19 = mne.time_frequency.read_tfrs('S19_power_s-tfr.h5')
power_s_19_1 = power_s_19[0]
power_s_18 = mne.time_frequency.read_tfrs('S18_power_s-tfr.h5')
power_s_18_1 = power_s_18[0]
power_s_17 = mne.time_frequency.read_tfrs('S17_power_s-tfr.h5')
power_s_17_1 = power_s_17[0]
power_s_16 = mne.time_frequency.read_tfrs('S16_power_s-tfr.h5')
power_s_16_1 = power_s_16[0]
power_s_15 = mne.time_frequency.read_tfrs('S15_power_s-tfr.h5')
power_s_15_1 = power_s_15[0]
power_s_14 = mne.time_frequency.read_tfrs('S14_power_s-tfr.h5')
power_s_14_1 = power_s_14[0]
power_s_13 = mne.time_frequency.read_tfrs('S13_power_s-tfr.h5')
power_s_13_1 = power_s_13[0]
power_s_12 = mne.time_frequency.read_tfrs('S12_power_s-tfr.h5')
power_s_12_1 = power_s_12[0]
power_s_11 = mne.time_frequency.read_tfrs('S11_power_s-tfr.h5')
power_s_11_1 = power_s_11[0]
power_s_10 = mne.time_frequency.read_tfrs('S10_power_s-tfr.h5')
power_s_10_1 = power_s_10[0]
power_s_9 = mne.time_frequency.read_tfrs('S9_power_s-tfr.h5')
power_s_9_1 = power_s_9[0]
power_s_8 = mne.time_frequency.read_tfrs('S8_power_s-tfr.h5')
power_s_8_1 = power_s_8[0]
power_s_7 = mne.time_frequency.read_tfrs('S7_power_s-tfr.h5')
power_s_7_1 = power_s_7[0]
power_s_6 = mne.time_frequency.read_tfrs('S6_power_s-tfr.h5')
power_s_6_1 = power_s_6[0]
power_s_5 = mne.time_frequency.read_tfrs('S5_power_s-tfr.h5')
power_s_5_1 = power_s_5[0]
power_s_4 = mne.time_frequency.read_tfrs('S4_power_s-tfr.h5')
power_s_4_1 = power_s_4[0]
power_s_3 = mne.time_frequency.read_tfrs('S3_power_s-tfr.h5')
power_s_3_1 = power_s_3[0]

power_t_28 = mne.time_frequency.read_tfrs('S28_power_t-tfr.h5')
power_t_28_1 = power_t_28[0]
power_t_27 = mne.time_frequency.read_tfrs('S27_power_t-tfr.h5')
power_t_27_1 = power_t_27[0]
power_t_26 = mne.time_frequency.read_tfrs('S26_power_t-tfr.h5')
power_t_26_1 = power_t_26[0]
power_t_25 = mne.time_frequency.read_tfrs('S25_power_t-tfr.h5')
power_t_25_1 = power_t_25[0]
power_t_24 = mne.time_frequency.read_tfrs('S24_power_t-tfr.h5')
power_t_24_1 = power_t_24[0]
power_t_23 = mne.time_frequency.read_tfrs('S23_power_t-tfr.h5')
power_t_23_1 = power_t_23[0]
power_t_22 = mne.time_frequency.read_tfrs('S22_power_t-tfr.h5')
power_t_22_1 = power_t_22[0]
power_t_21 = mne.time_frequency.read_tfrs('S21_power_t-tfr.h5')
power_t_21_1 = power_t_21[0]
power_t_20 = mne.time_frequency.read_tfrs('S20_power_t-tfr.h5')
power_t_20_1 = power_t_20[0]
power_t_19 = mne.time_frequency.read_tfrs('S19_power_t-tfr.h5')
power_t_19_1 = power_t_19[0]
power_t_18 = mne.time_frequency.read_tfrs('S18_power_t-tfr.h5')
power_t_18_1 = power_t_18[0]
power_t_17 = mne.time_frequency.read_tfrs('S17_power_t-tfr.h5')
power_t_17_1 = power_t_17[0]
power_t_16 = mne.time_frequency.read_tfrs('S16_power_t-tfr.h5')
power_t_16_1 = power_t_16[0]
power_t_15 = mne.time_frequency.read_tfrs('S15_power_t-tfr.h5')
power_t_15_1 = power_t_15[0]
power_t_14 = mne.time_frequency.read_tfrs('S14_power_t-tfr.h5')
power_t_14_1 = power_t_14[0]
power_t_13 = mne.time_frequency.read_tfrs('S13_power_t-tfr.h5')
power_t_13_1 = power_t_13[0]
power_t_12 = mne.time_frequency.read_tfrs('S12_power_t-tfr.h5')
power_t_12_1 = power_t_12[0]
power_t_11 = mne.time_frequency.read_tfrs('S11_power_t-tfr.h5')
power_t_11_1 = power_t_11[0]
power_t_10 = mne.time_frequency.read_tfrs('S10_power_t-tfr.h5')
power_t_10_1 = power_t_10[0]
power_t_9 = mne.time_frequency.read_tfrs('S9_power_t-tfr.h5')
power_t_9_1 = power_t_9[0]
power_t_8 = mne.time_frequency.read_tfrs('S8_power_t-tfr.h5')
power_t_8_1 = power_t_8[0]
power_t_7 = mne.time_frequency.read_tfrs('S7_power_t-tfr.h5')
power_t_7_1 = power_t_7[0]
power_t_6 = mne.time_frequency.read_tfrs('S6_power_t-tfr.h5')
power_t_6_1 = power_t_6[0]
power_t_5 = mne.time_frequency.read_tfrs('S5_power_t-tfr.h5')
power_t_5_1 = power_t_5[0]
power_t_4 = mne.time_frequency.read_tfrs('S4_power_t-tfr.h5')
power_t_4_1 = power_t_4[0]
power_t_3 = mne.time_frequency.read_tfrs('S3_power_t-tfr.h5')
power_t_3_1 = power_t_3[0]


# %% FIRST GATHERING

######### AUTOMATIC VERSION OF GATHERING
power_s_list = power_ls
power_t_list = power_lt

i=0
power_t_data = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
power_s_data = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]
for i in range(26):
    power_t_data[i] = power_lt[i].data
    power_s_data[i] = power_lt[i].data
    #a = power_lt[i].data
    #power_t_data.append(a)
    #b = power_ls[i].data
    #power_s_data.append(b)
    i+=1

type(power_t_data)
c = np.array(power_t_data)
type(c)
np.shape(c)

#########  MANUAL GATHERING
power_s_list = [power_s_28_1, power_s_27_1, power_s_26_1, power_s_25_1, power_s_24_1, 
                power_s_23_1, power_s_22_1, power_s_21_1, power_s_20_1, power_s_19_1, 
                power_s_18_1, power_s_17_1, power_s_16_1, power_s_15_1, power_s_14_1,
                power_s_13_1, power_s_12_1, power_s_11_1, power_s_10_1, power_s_9_1, 
                power_s_8_1, power_s_7_1, power_s_6_1, power_s_5_1, power_s_4_1, power_s_3_1]

power_s_data = [power_s_28_1.data, power_s_27_1.data, power_s_26_1.data, power_s_25_1.data, power_s_24_1.data, 
                power_s_23_1.data, power_s_22_1.data, power_s_21_1.data, power_s_20_1.data, power_s_19_1.data, 
                power_s_18_1.data, power_s_17_1.data, power_s_16_1.data, power_s_15_1.data, power_s_14_1.data,
                power_s_13_1.data, power_s_12_1.data, power_s_11_1.data, power_s_10_1.data, power_s_9_1.data, 
                power_s_8_1.data, power_s_7_1.data, power_s_6_1.data, power_s_5_1.data, power_s_4_1.data, power_s_3_1.data]

power_t_list = [power_t_28_1, power_t_27_1, power_t_26_1, power_t_25_1, power_t_24_1, 
                power_t_23_1, power_t_22_1, power_t_21_1, power_t_20_1, power_t_19_1, 
                power_t_18_1, power_t_17_1, power_t_16_1, power_t_15_1, power_t_14_1,
                power_t_13_1, power_t_12_1, power_t_11_1, power_t_10_1, power_t_9_1, 
                power_t_8_1, power_t_7_1, power_t_6_1, power_t_5_1, power_t_4_1, power_t_3_1]

power_t_data = [power_t_28_1.data, power_t_27_1.data, power_t_26_1.data, power_t_25_1.data, power_t_24_1.data, 
                power_t_23_1.data, power_t_22_1.data, power_t_21_1.data, power_t_20_1.data, power_t_19_1.data, 
                power_t_18_1.data, power_t_17_1.data, power_t_16_1.data, power_t_15_1.data, power_t_14_1.data,
                power_t_13_1.data, power_t_12_1.data, power_t_11_1.data, power_t_10_1.data, power_t_9_1.data, 
                power_t_8_1.data, power_t_7_1.data, power_t_6_1.data, power_t_5_1.data, power_t_4_1.data, power_t_3_1.data]

power_data = [power_t_28_1.data, power_t_27_1.data, power_t_26_1.data, power_t_25_1.data, power_t_24_1.data, 
                power_t_23_1.data, power_t_22_1.data, power_t_21_1.data, power_t_20_1.data, power_t_19_1.data, 
                power_t_18_1.data, power_t_17_1.data, power_t_16_1.data, power_t_15_1.data, power_t_14_1.data,
                power_t_13_1.data, power_t_12_1.data, power_t_11_1.data, power_t_10_1.data, power_t_9_1.data, 
                power_t_8_1.data, power_t_7_1.data, power_t_6_1.data, power_t_5_1.data, power_t_4_1.data, power_t_3_1.data,
                power_t_28_1.data, power_t_27_1.data, power_t_26_1.data, power_t_25_1.data, power_t_24_1.data, 
                                power_t_23_1.data, power_t_22_1.data, power_t_21_1.data, power_t_20_1.data, power_t_19_1.data, 
                                power_t_18_1.data, power_t_17_1.data, power_t_16_1.data, power_t_15_1.data, power_t_14_1.data,
                                power_t_13_1.data, power_t_12_1.data, power_t_11_1.data, power_t_10_1.data, power_t_9_1.data, 
                                power_t_8_1.data, power_t_7_1.data, power_t_6_1.data, power_t_5_1.data, power_t_4_1.data, power_t_3_1.data]

# %% CROPING THE FILE

i = 0 
for i in range(len(power_t_list)):
    power_t_list[i].crop(0,4)
    i+= 1

i = 0 
for i in range(len(power_s_list)):
    power_s_list[i].crop(0,4)
    i+= 1
    
i = 0 
for i in range(len(power_t_list)):
    power_t_data[i] = power_t_list[i].data
    i+= 1
    
i = 0 
for i in range(len(power_s_list)):
    power_s_data[i] = power_s_list[i].data
    i+= 1

# %% FOOOF normalization - WORKS

file_path = 'G:\\WS'

#### INITIATION
subj = 0
ch = 0
freq_range = [1, 80]
fm = FOOOF()
list_s_ped =   [1, 2, 3, 4, 5, 6, 7,8 , 9, 10, 1, 2, 3, 4, 5, 6, 7,8 , 9, 10, 1, 2, 3, 4, 5,0 ]
len(list_s_ped)
list_s_aper =  [1, 2, 3, 4, 5, 6, 7,8 , 9, 10, 1, 2, 3, 4, 5, 6, 7,8 , 9, 10, 1, 2, 3, 4, 5,0 ]
len(list_s_aper)
list_t2_ped =   [1, 2, 3, 4, 5, 6, 7,8 , 9, 10, 1, 2, 3, 4, 5, 6, 7,8 , 9, 10, 1, 2, 3, 4, 5,0 ]
len(list_t_ped)
list_t2_aper =  [1, 2, 3, 4, 5, 6, 7,8 , 9, 10, 1, 2, 3, 4, 5, 6, 7,8 , 9, 10, 1, 2, 3, 4, 5,0 ]
len(list_t_aper)

########## SPATIAL CONDITION
for subj in range(len(power_s_data)):    
    spectrum_peak = np.array([])    #Here we story everything for 204 channel in 1 subject
    spectrum_aper = np.array([])
    power_s = power_s_list[subj]
    a = power_s.freqs
    b = power_s.data
    a.shape
    b.shape
    f = np.mean(b, axis=2)
    type(f)
    f.shape
    freqs = a 
    freqs.shape             # 30 freqs points
    spectrum = f
    spectrum.shape          # 204 ch for 30 freqs powers
    
    for ch in np.arange(204):       
        spec = spectrum[ch, :]
        spec.shape
        fm.fit(freqs, spec, freq_range)
#        fm.save('FOOOF_sub{}_results_S_{}'.format(subj, i), file_path, save_results=True, save_settings=True, save_data=True)
        init_ap_fit = gen_aperiodic(fm.freqs, fm._robust_ap_fit(fm.freqs, fm.power_spectrum))
        type(init_ap_fit)
        np.shape(init_ap_fit)
        init_flat_spec = fm.power_spectrum - init_ap_fit
        type(init_flat_spec)
        np.shape(init_flat_spec)

        spectrum_peak = np.append(spectrum_peak, init_flat_spec.T)
        spectrum_aper = np.append(spectrum_aper, init_ap_fit.T)
        spectrum_peak.size
        spectrum_aper.size
        spectrum_peak.shape
    
        spectrum_peak = np.reshape(spectrum_peak,[ch+1,len(freqs)]) #( 204, 30) for one subject
        spectrum_aper = np.reshape(spectrum_aper,[ch+1,len(freqs)]) #( 204, 30) for one subject
        spectrum_peak.shape
        spectrum_aper.shape

        ch += 1
    list_s_ped[subj] = spectrum_peak
    list_s_aper[subj] = spectrum_aper
    subj += 1

########## TEMPORAL CONDITION
for subj in range(len(power_t_data)):    
    spectrum_peak_t = np.array([])    #Here we story everything for 204 channel in 1 subject
    spectrum_aper_t = np.array([])
    power_t = power_t_list[subj]
    a = power_t.freqs
    b = power_t.data
    a.shape
    b.shape
    f = np.mean(b, axis=2)
    type(f)
    f.shape
    freqs = a 
    freqs.shape             # 30 freqs points
    spectrum = f
    spectrum.shape          # 204 ch for 30 freqs powers
    
    for ch in np.arange(204):       
        spec = spectrum[ch, :]
        spec.shape
        fm.fit(freqs, spec, freq_range)
#        fm.save('FOOOF_sub{}_results_T_{}'.format(subj, i), file_path, save_results=True, save_settings=True, save_data=True)
        init_ap_fit = gen_aperiodic(fm.freqs, fm._robust_ap_fit(fm.freqs, fm.power_spectrum))
        type(init_ap_fit)
        np.shape(init_ap_fit)
        init_flat_spec = fm.power_spectrum - init_ap_fit
        type(init_flat_spec)
        np.shape(init_flat_spec)

        spectrum_peak_t = np.append(spectrum_peak_t, init_flat_spec.T)
        spectrum_aper_t = np.append(spectrum_aper_t, init_ap_fit.T)
        spectrum_peak_t.size
        spectrum_aper_t.size
        spectrum_peak_t.shape
    
        spectrum_peak_t = np.reshape(spectrum_peak_t,[ch+1,len(freqs)]) #( 204, 30) for one subject
        spectrum_aper_t = np.reshape(spectrum_aper_t,[ch+1,len(freqs)]) #( 204, 30) for one subject
        spectrum_peak_t.shape
        spectrum_aper_t.shape

        ch += 1
    list_t_ped[subj] = spectrum_peak_t
    list_t_aper[subj] = spectrum_aper_t
    subj += 1


# %% PLOTTING THE FOOOF TO CHECK
freq_range = [1, 80]
plt_log = False

#Individual
subj = 1
power_t = power_t_list[subj]
a = list_t_ped[1]
freqs = power_t.freqs
plot_spectrum(fm.freqs, a.T, plt_log)

#All Subjects
for subj in range(len(power_t_data)):
    power_t = power_t_list[subj]
    freqs = power_t.freqs
    a = list_t_ped[subj]
    plot_spectrum(fm.freqs, a.T, plt_log)
    subj +=1

#One Subjects and particular 
subj = 1
n_channels = 15

power_t = power_t_list[subj]
freqs = power_t.freqs
b = list_t_ped[subj]
for ch in np.arange(n_channels):  
    a = b[ch,:]
    plot_spectrum(fm.freqs, a.T, plt_log)
    ch+=1

    
#All Subjects and particular channel
len(power_t_data)

for subj in range(2):
    power_t = power_t_list[subj]
    freqs = power_t.freqs
    b = list_t_ped[subj]

    for ch in np.arange(10):  
        a = b[ch,:]
        plot_spectrum(fm.freqs, a.T, plt_log)
        ch+=1

    subj +=1
    
#COMBINED POWER SPECTRA
subj = 1
power_s = power_s_list[subj]
power_t = power_t_list[subj]
freqs = power_t.freqs
freqs.shape

a = list_t_ped[subj]
b = list_s_ped[subj]
a.shape
b.shape
p_s = np.mean(a, axis=0)
p_t = np.mean(b, axis=0)
p_s.shape
p_t.shape

labels = ['p_s', 'p_t']
plot_spectra(freqs, [p_t, p_s], log_powers=False,  labels=labels)

# %% SAVER
#NPY SAVER
a = np.array(list_t_ped)
type(a)
a.shape
b = np.array(list_t_aper)
type(b)
b.shape
c = np.array(list_s_ped)
type(c)
c.shape
d = np.array(list_s_aper)
type(d)
d.shape

np.save(file='list_t_ped.npy', arr=a)
np.save(file='list_t_aper.npy', arr=b)
np.save(file='list_s_ped.npy', arr=c)
np.save(file='list_s_aper.npy', arr=d)

# %% FOOOF OBJECT LOADER
#NPY LOAD
list_t_ped_np = np.load(file='list_t_ped.npy')
list_t_aper_np = np.load(file='list_t_aper.npy')
list_s_ped_np = np.load(file='list_s_ped.npy')
list_s_aper_np = np.load(file='list_s_aper.npy')

type(list_t_ped_np)
type(list_t_aper_np)
type(list_s_ped_np)
type(list_s_aper_np)
len(list_t_ped_np)
len(list_t_aper_np)
len(list_s_ped_np)
len(list_s_aper_np)

list_t_ped_np.shape
list_t_aper_np.shape
list_s_ped_np.shape
list_s_aper_np.shape
    
# %% STATISTICS

####### INPUT
list_s_ped = list_s_ped_np
list_t_ped = list_t_ped_np

psa = list_s_ped_np #NOW THE DATA AFTER NORMALIZATION TAKES SHAPE - SUB, CH, n_freqs, n_times 
pta = list_t_ped_np # 26 for (204, 30, 201)

psa.shape
pta.shape
a_s = psa
a_t = pta

####### DOWNSAMPLING
# axis = 
# a_s = np.mean(psa, axis)
# a_t = np.mean(pta, axis)

a_s = np.transpose(psa, (0,2,1))
a_t = np.transpose(pta, (0,2,1))
a_s.shape
a_t.shape


####### STATISTICS
threshold = 2.0
n_permutations = 2000
epochs = epochs_s_28
info = epochs.info
adj, ch_names = mne.channels.find_ch_adjacency(info, ch_type= 'grad')

# Non-parametric cluster-level paired t-test for spatio-temporal data.
obj = a_s - a_t
T_obs, clusters, cluster_p_values, H0  = mne.stats.spatio_temporal_cluster_1samp_test(obj, 
                                             out_type='mask', adjacency=adj, 
                                             n_permutations=n_permutations,
                                             threshold=threshold, tail=0)

type(T_obs) 
type(clusters)
type(cluster_p_values) 
type(H0)
np.shape(T_obs) 
np.shape(clusters)
np.shape(cluster_p_values) 
np.shape(H0)

plt.imshow(T_obs, aspect='auto', origin='lower', cmap='gray', vmin=None, vmax=-threshold) #freq not just linear, but as index

####### PLOTING MULTI CHANNELS

plt.plot(a_s[:,:,1].T) #Channel representation over frequency 
plt.plot(obj[:,:,1].T) #Channel representation over frequency 
# freqs = power_ls[1].freqs
# plt.plot([a_s[:,:,1].T, a_t[:,:,1].T], freqs)  #TWO channel representations over frequency 
plt.show()


####### CHECKING WITH SIGNIFICANT LEVEL
p_accept = 0.01
good_cluster_inds = np.where(cluster_p_values < p_accept)[0]
print(good_cluster_inds)
len(good_cluster_inds)
print(cluster_p_values)


####### PLOTING SIGNIFICANT CLUSTERS
plt.subplot(2, 1, 1)
T_obs_plot = np.nan * np.ones_like(T_obs)

for c, p_val in zip(clusters, cluster_p_values):
    if p_val <= 0.05:
         T_obs_plot[c] = T_obs[c]
         
plt.imshow(T_obs, aspect='auto', origin='lower', cmap='gray') #freq not just linear, but as index
plt.colorbar(True)
plt.imshow(T_obs_plot,aspect='auto', origin='lower', cmap='RdBu_r')
plt.show()

plt.xlabel('Time (ms)')
plt.ylabel('Frequency (Hz)')
plt.title('Induced power')
    