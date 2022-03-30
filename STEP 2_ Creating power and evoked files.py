# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 17:35:28 2022

@author: Nikita O
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 17:21:38 2022

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

from mne.cov import compute_covariance
from mne.datasets import somato
from mne.time_frequency import csd_morlet
from mne.beamformer import (make_dics, apply_dics_csd, make_lcmv,
                            apply_lcmv_cov)
from mne.minimum_norm import (make_inverse_operator, apply_inverse_cov)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# %% WORKING WITH EPOCHS

epochs_s_28 = mne.read_epochs('S28_full_S_epochs-epo.fif', preload=False)
epochs_t_28 = mne.read_epochs('S28_full_T_epochs-epo.fif', preload=False)
epochs_s_27 = mne.read_epochs('S27_full_S_epochs-epo.fif', preload=False)
epochs_t_27 = mne.read_epochs('S27_full_T_epochs-epo.fif', preload=False)
epochs_s_26 = mne.read_epochs('S26_full_S_epochs-epo.fif', preload=False)
epochs_t_26 = mne.read_epochs('S26_full_T_epochs-epo.fif', preload=False)
epochs_s_25 = mne.read_epochs('S25_full_S_epochs-epo.fif', preload=False)
epochs_t_25 = mne.read_epochs('S25_full_T_epochs-epo.fif', preload=False)
epochs_s_23 = mne.read_epochs('S23_full_S_epochs-epo.fif', preload=False)
epochs_t_23 = mne.read_epochs('S23_full_T_epochs-epo.fif', preload=False)
epochs_s_21 = mne.read_epochs('S21_full_S_epochs-epo.fif', preload=False)
epochs_t_21 = mne.read_epochs('S21_full_T_epochs-epo.fif', preload=False)
epochs_s_20 = mne.read_epochs('S20_full_S_epochs-epo.fif', preload=False)
epochs_t_20 = mne.read_epochs('S20_full_T_epochs-epo.fif', preload=False)
epochs_s_19 = mne.read_epochs('S19_full_S_epochs-epo.fif', preload=False)
epochs_t_19 = mne.read_epochs('S19_full_T_epochs-epo.fif', preload=False)
epochs_s_18 = mne.read_epochs('S18_full_S_epochs-epo.fif', preload=False)
epochs_t_18 = mne.read_epochs('S18_full_T_epochs-epo.fif', preload=False)
epochs_s_14 = mne.read_epochs('S14_full_S_epochs-epo.fif', preload=False)
epochs_t_14 = mne.read_epochs('S14_full_T_epochs-epo.fif', preload=False)
epochs_s_11 = mne.read_epochs('S11_full_S_epochs-epo.fif', preload=False)
epochs_t_11 = mne.read_epochs('S11_full_T_epochs-epo.fif', preload=False)
epochs_s_10 = mne.read_epochs('S10_full_S_epochs-epo.fif', preload=False)
epochs_t_10 = mne.read_epochs('S10_full_T_epochs-epo.fif', preload=False)
epochs_s_9 = mne.read_epochs('S9_full_S_epochs-epo.fif', preload=False)
epochs_t_9 = mne.read_epochs('S9_full_T_epochs-epo.fif', preload=False)
epochs_s_7 = mne.read_epochs('S7_full_S_epochs-epo.fif', preload=False)
epochs_t_7 = mne.read_epochs('S7_full_T_epochs-epo.fif', preload=False)
epochs_s_6 = mne.read_epochs('S6_full_S_epochs-epo.fif', preload=False)
epochs_t_6 = mne.read_epochs('S6_full_T_epochs-epo.fif', preload=False)
epochs_s_4 = mne.read_epochs('S4_full_S_epochs-epo.fif', preload=False)
epochs_t_4 = mne.read_epochs('S4_full_T_epochs-epo.fif', preload=False)
epochs_s_3 = mne.read_epochs('S3_full_S_epochs-epo.fif', preload=False)
epochs_t_3 = mne.read_epochs('S3_full_T_epochs-epo.fif', preload=False)
epochs_s_5 = mne.read_epochs('S5_full_S_epochs-epo.fif', preload=False)
epochs_t_5 = mne.read_epochs('S5_full_T_epochs-epo.fif', preload=False)
epochs_s_8 = mne.read_epochs('S8_full_S_epochs-epo.fif', preload=False)
epochs_t_8 = mne.read_epochs('S8_full_T_epochs-epo.fif', preload=False)
epochs_s_12 = mne.read_epochs('S12_full_S_epochs-epo.fif', preload=False)
epochs_t_12 = mne.read_epochs('S12_full_T_epochs-epo.fif', preload=False)
epochs_s_13 = mne.read_epochs('S13_full_S_epochs-epo.fif', preload=False)
epochs_t_13 = mne.read_epochs('S13_full_T_epochs-epo.fif', preload=False)
epochs_s_15 = mne.read_epochs('S15_full_S_epochs-epo.fif', preload=False)
epochs_t_15 = mne.read_epochs('S15_full_T_epochs-epo.fif', preload=False)
epochs_s_16 = mne.read_epochs('S16_full_S_epochs-epo.fif', preload=False)
epochs_t_16 = mne.read_epochs('S16_full_T_epochs-epo.fif', preload=False)
epochs_s_17 = mne.read_epochs('S17_full_S_epochs-epo.fif', preload=False)
epochs_t_17 = mne.read_epochs('S17_full_T_epochs-epo.fif', preload=False)
epochs_s_22 = mne.read_epochs('S22_full_S_epochs-epo.fif', preload=False)
epochs_t_22 = mne.read_epochs('S22_full_T_epochs-epo.fif', preload=False)
epochs_s_24 = mne.read_epochs('S24_full_S_epochs-epo.fif', preload=False)
epochs_t_24 = mne.read_epochs('S24_full_T_epochs-epo.fif', preload=False)


epochs_s = epochs_s_28
epochs_t = epochs_t_28
index = 28


#Creating evoked object (average epochs)
spat_evoked = epochs_s.average()
temp_evoked = epochs_t.average()


#LINE with topomap
# #COMPARISON PLOT
# mne.viz.plot_compare_evokeds(dict(spatial=spat_evoked, temporal=temp_evoked),
#                               legend='upper left', show_sensors='upper right')
# evoked_diff = mne.combine_evoked([spat_evoked, temp_evoked], weights=[1, -1])
# evoked_diff.plot_topo(color='r', legend=False)
#grand_average = mne.grand_average([spat_evoked, temp_evoked])



# %% MERGED PSD PICTURE
epochs_s = epochs_s_26
epochs_t = epochs_t_26
index = 26

#Two plots in one figure

fig, ax = plt.subplots(2)
epochs_s.plot_psd(tmin=0, tmax=4, fmin = 1, fmax=80, average=True, ax=ax[0], normalization = 'length') #full / spatial_color = True
ax[0].set_title('PSD for spatial information of participant  {}'.format(index))
ax[0].set_ylabel('(fT/cm)^2/Hz (dB)')
epochs_t.plot_psd(tmin=0, tmax=4, fmin = 1, fmax=80, average=True, ax=ax[1], normalization = 'length') #full / spatial_color = True
ax[1].set_title('PSD for temporal information of participant  {}'.format(index))
ax[1].set_ylabel('(fT/cm)^2/Hz (dB)')
ax[1].set_xlabel('Frequency (Hz)')
fig.set_tight_layout(True)
#fig.set_figwidth(40)
#fig.set_figheight(40)
plt.show()

#Two plots merged
gs = gridspec.GridSpec(2,1)
plt.figure()
ax = plt.axes()
epochs_s.plot_psd(ax=ax, tmin=0, tmax=4, fmin = 1, fmax=80,  average=True, color= 'r')
epochs_t.plot_psd(ax=ax, tmin=0, tmax=4, fmin = 1, fmax=80,  average=True, color= 'b')
plt.show()

# %% PSD part

#frequencies =    np.arange(4, 80, 2)
epochs_s = epochs_s_3
epochs_t = epochs_t_3
index = 3

spat_evoked = epochs_s.average()
temp_evoked = epochs_t.average()

a = np.log10(4)
b = np.log10(80)
frequencies = np.logspace(a,b, num=30)

#tic = timeit.default_timer()


power_s, itc_s = mne.time_frequency.tfr_morlet(epochs_s, n_cycles=5, return_itc=True,
                                          freqs=frequencies, decim=20)
power_t, itc_t = mne.time_frequency.tfr_morlet(epochs_t, n_cycles=5, return_itc=True,
                                      freqs=frequencies, decim=20)

power_s.save('S{}_power_s-tfr.h5'.format(index))
power_t.save('S{}_power_t-tfr.h5'.format(index))
itc_s.save('S{}_itc_s-tfr.h5'.format(index))
itc_t.save('S{}_itc_t-tfr.h5'.format(index))
spat_evoked.save('S{}_evoked_s-ave.fif'.format(index))
temp_evoked.save('S{}_evoked_t-ave.fif'.format(index))


#PLOTING
power_s.plot_topo(baseline=(None,None), mode = 'logratio', title =' average power temporal')
power_t.plot_topo(baseline=(None,None), mode = 'logratio', title =' average power temporal')


# %% CREATING EVOKED FOR ALL

spat_evoked_28 = epochs_s_28.average()
spat_evoked_27 = epochs_s_27.average()
spat_evoked_26 = epochs_s_26.average()
spat_evoked_25 = epochs_s_25.average()
spat_evoked_24 = epochs_s_24.average()
spat_evoked_23 = epochs_s_23.average()
spat_evoked_22 = epochs_s_22.average()
spat_evoked_21 = epochs_s_21.average()
spat_evoked_20 = epochs_s_20.average()
spat_evoked_19 = epochs_s_19.average()
spat_evoked_18 = epochs_s_18.average()
spat_evoked_17 = epochs_s_17.average()
spat_evoked_16 = epochs_s_16.average()
spat_evoked_15 = epochs_s_15.average()
spat_evoked_14 = epochs_s_14.average()
spat_evoked_13 = epochs_s_13.average()
spat_evoked_12 = epochs_s_12.average()
spat_evoked_11 = epochs_s_11.average()
spat_evoked_10 = epochs_s_10.average()
spat_evoked_9 = epochs_s_9.average()
spat_evoked_8 = epochs_s_8.average()
spat_evoked_7 = epochs_s_7.average()
spat_evoked_6 = epochs_s_6.average()
spat_evoked_5 = epochs_s_5.average()
spat_evoked_4 = epochs_s_4.average()
spat_evoked_3 = epochs_s_3.average()

temp_evoked_28 = epochs_t_28.average()
temp_evoked_27 = epochs_t_27.average()
temp_evoked_26 = epochs_t_26.average()
temp_evoked_25 = epochs_t_25.average()
temp_evoked_24 = epochs_t_24.average()
temp_evoked_23 = epochs_t_23.average()
temp_evoked_22 = epochs_t_22.average()
temp_evoked_21 = epochs_t_21.average()
temp_evoked_20 = epochs_t_20.average()
temp_evoked_19 = epochs_t_19.average()
temp_evoked_18 = epochs_t_18.average()
temp_evoked_17 = epochs_t_17.average()
temp_evoked_16 = epochs_t_16.average()
temp_evoked_15 = epochs_t_15.average()
temp_evoked_14 = epochs_t_14.average()
temp_evoked_13 = epochs_t_13.average()
temp_evoked_12 = epochs_t_12.average()
temp_evoked_11 = epochs_t_11.average()
temp_evoked_10 = epochs_t_10.average()
temp_evoked_9 = epochs_t_9.average()
temp_evoked_8 = epochs_t_8.average()
temp_evoked_7 = epochs_t_7.average()
temp_evoked_6 = epochs_t_6.average()
temp_evoked_5 = epochs_t_5.average()
temp_evoked_4 = epochs_t_4.average()
temp_evoked_3 = epochs_t_3.average()

evoked_s_list = [epochs_s_28, epochs_s_27, epochs_s_26, epochs_s_25, epochs_s_24, 
                 epochs_s_23, epochs_s_22, epochs_s_21, epochs_s_20, epochs_s_19, 
                 epochs_s_18, epochs_s_17, epochs_s_16, epochs_s_15, epochs_s_14,
                 epochs_s_13, epochs_s_12, epochs_s_11, epochs_s_10, epochs_s_9,
                 epochs_s_8, epochs_s_7, epochs_s_6, epochs_s_5, epochs_s_4, epochs_s_3]

evoked_t_list = [epochs_t_28, epochs_t_27, epochs_t_26, epochs_t_25, epochs_t_24, 
                 epochs_t_23, epochs_t_22, epochs_t_21, epochs_t_20, epochs_t_19, 
                 epochs_t_18, epochs_t_17, epochs_t_16, epochs_t_15, epochs_t_14,
                 epochs_t_13, epochs_t_12, epochs_t_11, epochs_t_10, epochs_t_9,
                 epochs_t_8, epochs_t_7, epochs_t_6, epochs_t_5, epochs_t_4, epochs_t_3]



# %% DATA LOADING

#LOADING EVOKES
temp_evoked_28 = mne.read_evokeds('S28_evoked_t-ave.fif')
te_28 = temp_evoked_28[0]
temp_evoked_27 = mne.read_evokeds('S27_evoked_t-ave.fif')
te_27 = temp_evoked_27[0]
temp_evoked_26 = mne.read_evokeds('S26_evoked_t-ave.fif')
te_26 = temp_evoked_26[0]
temp_evoked_25 = mne.read_evokeds('S25_evoked_t-ave.fif')
te_25 = temp_evoked_25[0]
temp_evoked_24 = mne.read_evokeds('S24_evoked_t-ave.fif')
te_24 = temp_evoked_24[0]
temp_evoked_23 = mne.read_evokeds('S23_evoked_t-ave.fif')
te_23 = temp_evoked_23[0]
temp_evoked_22 = mne.read_evokeds('S22_evoked_t-ave.fif')
te_22 = temp_evoked_22[0]
temp_evoked_21 = mne.read_evokeds('S21_evoked_t-ave.fif')
te_21 = temp_evoked_21[0]
temp_evoked_20 = mne.read_evokeds('S20_evoked_t-ave.fif')
te_20 = temp_evoked_20[0]
temp_evoked_19 = mne.read_evokeds('S19_evoked_t-ave.fif')
te_19 = temp_evoked_19[0]
temp_evoked_18 = mne.read_evokeds('S18_evoked_t-ave.fif')
te_18 = temp_evoked_18[0]
temp_evoked_17 = mne.read_evokeds('S17_evoked_t-ave.fif')
te_17 = temp_evoked_17[0]
temp_evoked_16 = mne.read_evokeds('S16_evoked_t-ave.fif')
te_16 = temp_evoked_16[0]
temp_evoked_15 = mne.read_evokeds('S15_evoked_t-ave.fif')
te_15 = temp_evoked_15[0]
temp_evoked_14 = mne.read_evokeds('S14_evoked_t-ave.fif')
te_14 = temp_evoked_14[0]
temp_evoked_13 = mne.read_evokeds('S13_evoked_t-ave.fif')
te_13 = temp_evoked_13[0]
temp_evoked_12 = mne.read_evokeds('S12_evoked_t-ave.fif')
te_12 = temp_evoked_12[0]
temp_evoked_11 = mne.read_evokeds('S11_evoked_t-ave.fif')
te_11 = temp_evoked_11[0]
temp_evoked_10 = mne.read_evokeds('S10_evoked_t-ave.fif')
te_10 = temp_evoked_10[0]
temp_evoked_9 = mne.read_evokeds('S9_evoked_t-ave.fif')
te_9 = temp_evoked_9[0]
temp_evoked_8 = mne.read_evokeds('S8_evoked_t-ave.fif')
te_8 = temp_evoked_8[0]
temp_evoked_7 = mne.read_evokeds('S7_evoked_t-ave.fif')
te_7 = temp_evoked_7[0]
temp_evoked_6 = mne.read_evokeds('S6_evoked_t-ave.fif')
te_6 = temp_evoked_6[0]
temp_evoked_5 = mne.read_evokeds('S5_evoked_t-ave.fif')
te_5 = temp_evoked_5[0]
temp_evoked_4 = mne.read_evokeds('S4_evoked_t-ave.fif')
te_4 = temp_evoked_4[0]
temp_evoked_3 = mne.read_evokeds('S3_evoked_t-ave.fif')
te_3 = temp_evoked_3[0]

spat_evoked_28 = mne.read_evokeds('S28_evoked_s-ave.fif')
se_28 = spat_evoked_28[0]
spat_evoked_27 = mne.read_evokeds('S27_evoked_s-ave.fif')
se_27 = spat_evoked_27[0]
spat_evoked_26 = mne.read_evokeds('S26_evoked_s-ave.fif')
se_26 = spat_evoked_26[0]
spat_evoked_25 = mne.read_evokeds('S25_evoked_s-ave.fif')
se_25 = spat_evoked_25[0]
spat_evoked_24 = mne.read_evokeds('S24_evoked_s-ave.fif')
se_24 = spat_evoked_24[0]
spat_evoked_23 = mne.read_evokeds('S23_evoked_s-ave.fif')
se_23 = spat_evoked_23[0]
spat_evoked_22 = mne.read_evokeds('S22_evoked_s-ave.fif')
se_22 = spat_evoked_22[0]
spat_evoked_21 = mne.read_evokeds('S21_evoked_s-ave.fif')
se_21 = spat_evoked_21[0]
spat_evoked_20 = mne.read_evokeds('S20_evoked_s-ave.fif')
se_20 = spat_evoked_20[0]
spat_evoked_19 = mne.read_evokeds('S19_evoked_s-ave.fif')
se_19 = spat_evoked_19[0]
spat_evoked_18 = mne.read_evokeds('S18_evoked_s-ave.fif')
se_18 = spat_evoked_18[0]
spat_evoked_17 = mne.read_evokeds('S17_evoked_s-ave.fif')
se_17 = spat_evoked_17[0]
spat_evoked_16 = mne.read_evokeds('S16_evoked_s-ave.fif')
se_16 = spat_evoked_16[0]
spat_evoked_15 = mne.read_evokeds('S15_evoked_s-ave.fif')
se_15 = spat_evoked_15[0]
spat_evoked_14 = mne.read_evokeds('S14_evoked_s-ave.fif')
se_14 = spat_evoked_14[0]
spat_evoked_13 = mne.read_evokeds('S13_evoked_s-ave.fif')
se_13 = spat_evoked_13[0]
spat_evoked_12 = mne.read_evokeds('S12_evoked_s-ave.fif')
se_12 = spat_evoked_12[0]
spat_evoked_11 = mne.read_evokeds('S11_evoked_s-ave.fif')
se_11 = spat_evoked_11[0]
spat_evoked_10 = mne.read_evokeds('S10_evoked_s-ave.fif')
se_10 = spat_evoked_10[0]
spat_evoked_9a = mne.read_evokeds('S9_evoked_s-ave.fif')
se_9 = spat_evoked_9a[0]
spat_evoked_8 = mne.read_evokeds('S8_evoked_s-ave.fif')
se_8 = spat_evoked_8[0]
spat_evoked_7 = mne.read_evokeds('S7_evoked_s-ave.fif')
se_7 = spat_evoked_7[0]
spat_evoked_6 = mne.read_evokeds('S6_evoked_s-ave.fif')
se_6 = spat_evoked_6[0]
spat_evoked_5 = mne.read_evokeds('S5_evoked_s-ave.fif')
se_5 = spat_evoked_5[0]
spat_evoked_4 = mne.read_evokeds('S4_evoked_s-ave.fif')
se_4 = spat_evoked_4[0]
spat_evoked_3 = mne.read_evokeds('S3_evoked_s-ave.fif')
se_3 = spat_evoked_3[0]





    
