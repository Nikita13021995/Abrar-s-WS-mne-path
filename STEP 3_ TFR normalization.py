# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 17:27:53 2022

@author: Nikita O
"""
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 09:11:47 2022

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


# %% LOADING

epochs_s_28 = mne.read_epochs('S28_full_S_epochs-epo.fif', preload=False)
epochs_t_28 = mne.read_epochs('S28_full_T_epochs-epo.fif', preload=False)
epochs_s_27 = mne.read_epochs('S27_full_S_epochs-epo.fif', preload=False)
epochs_t_27 = mne.read_epochs('S27_full_T_epochs-epo.fif', preload=False)
epochs_s_26 = mne.read_epochs('S26_full_S_epochs-epo.fif', preload=False)
epochs_t_26 = mne.read_epochs('S26_full_T_epochs-epo.fif', preload=False)
epochs_s_25 = mne.read_epochs('S25_full_S_epochs-epo.fif', preload=False)
epochs_t_25 = mne.read_epochs('S25_full_T_epochs-epo.fif', preload=False)
epochs_t_24 = mne.read_epochs('S25_full_T_epochs-epo.fif', preload=False)
epochs_s_24 = mne.read_epochs('S23_full_S_epochs-epo.fif', preload=False)
epochs_s_23 = mne.read_epochs('S23_full_S_epochs-epo.fif', preload=False)
epochs_t_23 = mne.read_epochs('S23_full_T_epochs-epo.fif', preload=False)
epochs_s_22 = mne.read_epochs('S23_full_S_epochs-epo.fif', preload=False)
epochs_t_22 = mne.read_epochs('S23_full_T_epochs-epo.fif', preload=False)
epochs_s_21 = mne.read_epochs('S21_full_S_epochs-epo.fif', preload=False)
epochs_t_21 = mne.read_epochs('S21_full_T_epochs-epo.fif', preload=False)
epochs_s_20 = mne.read_epochs('S20_full_S_epochs-epo.fif', preload=False)
epochs_t_20 = mne.read_epochs('S20_full_T_epochs-epo.fif', preload=False)
epochs_s_19 = mne.read_epochs('S19_full_S_epochs-epo.fif', preload=False)
epochs_t_19 = mne.read_epochs('S19_full_T_epochs-epo.fif', preload=False)
epochs_s_18 = mne.read_epochs('S18_full_S_epochs-epo.fif', preload=False)
epochs_t_18 = mne.read_epochs('S18_full_T_epochs-epo.fif', preload=False)
epochs_s_17 = mne.read_epochs('S18_full_S_epochs-epo.fif', preload=False)
epochs_t_17 = mne.read_epochs('S18_full_T_epochs-epo.fif', preload=False)
epochs_s_16 = mne.read_epochs('S18_full_S_epochs-epo.fif', preload=False)
epochs_t_16 = mne.read_epochs('S18_full_T_epochs-epo.fif', preload=False)
epochs_s_15 = mne.read_epochs('S18_full_S_epochs-epo.fif', preload=False)
epochs_t_15 = mne.read_epochs('S18_full_T_epochs-epo.fif', preload=False)
epochs_s_14 = mne.read_epochs('S14_full_S_epochs-epo.fif', preload=False)
epochs_t_14 = mne.read_epochs('S14_full_T_epochs-epo.fif', preload=False)
epochs_s_13 = mne.read_epochs('S14_full_S_epochs-epo.fif', preload=False)
epochs_t_13 = mne.read_epochs('S14_full_T_epochs-epo.fif', preload=False)
epochs_s_12 = mne.read_epochs('S14_full_S_epochs-epo.fif', preload=False)
epochs_t_12 = mne.read_epochs('S14_full_T_epochs-epo.fif', preload=False)
epochs_s_11 = mne.read_epochs('S11_full_S_epochs-epo.fif', preload=False)
epochs_t_11 = mne.read_epochs('S11_full_T_epochs-epo.fif', preload=False)
epochs_s_10 = mne.read_epochs('S10_full_S_epochs-epo.fif', preload=False)
epochs_t_10 = mne.read_epochs('S10_full_T_epochs-epo.fif', preload=False)
epochs_s_9 = mne.read_epochs('S9_full_S_epochs-epo.fif', preload=False)
epochs_t_9 = mne.read_epochs('S9_full_T_epochs-epo.fif', preload=False)
epochs_s_8 = mne.read_epochs('S9_full_S_epochs-epo.fif', preload=False)
epochs_t_8 = mne.read_epochs('S9_full_T_epochs-epo.fif', preload=False)
epochs_s_7 = mne.read_epochs('S7_full_S_epochs-epo.fif', preload=False)
epochs_t_7 = mne.read_epochs('S7_full_T_epochs-epo.fif', preload=False)
epochs_s_6 = mne.read_epochs('S6_full_S_epochs-epo.fif', preload=False)
epochs_t_6 = mne.read_epochs('S6_full_T_epochs-epo.fif', preload=False)
epochs_s_5 = mne.read_epochs('S6_full_S_epochs-epo.fif', preload=False)
epochs_t_5 = mne.read_epochs('S6_full_T_epochs-epo.fif', preload=False)
epochs_s_4 = mne.read_epochs('S4_full_S_epochs-epo.fif', preload=False)
epochs_t_4 = mne.read_epochs('S4_full_T_epochs-epo.fif', preload=False)
epochs_s_3 = mne.read_epochs('S3_full_S_epochs-epo.fif', preload=False)
epochs_t_3 = mne.read_epochs('S3_full_T_epochs-epo.fif', preload=False)


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
spat_evoked_9 = mne.read_evokeds('S9_evoked_s-ave.fif')
se_9 = spat_evoked_9[0]
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

# %% GATHERING

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


evoked_t_list = [te_28, te_27, te_26, te_25, te_24, te_23, te_22, te_21, te_20, 
                  te_19, te_18, te_17, te_16, te_15, te_14, te_13, te_12, te_11, 
                  te_10, te_9, te_8, te_7, te_6, te_5, te_4, te_3]

evoked_s_list = [se_28, se_27, se_26, se_25, se_24, se_23, se_22, se_21, se_20, 
                  se_19, se_18, se_17, se_16, se_15, se_14, se_13, se_12, se_11, 
                  se_10, se_9, se_8, se_7, se_6, se_5, se_4, se_3]

evoked_mix = [te_28, te_27, te_26, te_25, te_24, te_23, te_22, te_21, te_20, 
                  te_19, te_18, te_17, te_16, te_15, te_14, te_13, te_12, te_11, 
                  te_10, te_9, te_8, te_7, te_6, te_5, te_4, te_3,se_28, se_27, se_26, se_25, se_24, se_23, se_22, se_21, se_20, 
                                    se_19, se_18, se_17, se_16, se_15, se_14, se_13, se_12, se_11, 
                                    se_10, se_9, se_8, se_7, se_6, se_5, se_4, se_3 ]

epochs_s_list = [epochs_s_28, epochs_s_27, epochs_s_26, epochs_s_25, epochs_s_24, 
                epochs_s_23, epochs_s_22, epochs_s_21, epochs_s_20, epochs_s_19, 
                epochs_s_18, epochs_s_17, epochs_s_16, epochs_s_15, epochs_s_14,
                epochs_s_13, epochs_s_12, epochs_s_11, epochs_s_10, epochs_s_9, 
                epochs_s_8, epochs_s_7, epochs_s_6, epochs_s_5, epochs_s_4, epochs_s_3]

epochs_t_list = [epochs_t_28, epochs_t_27, epochs_t_26, epochs_s_25, epochs_s_24, 
                epochs_t_23, epochs_t_22, epochs_t_21, epochs_s_20, epochs_s_19, 
                epochs_t_18, epochs_t_17, epochs_s_16, epochs_s_15, epochs_s_14,
                epochs_t_13, epochs_t_12, epochs_s_11, epochs_s_10, epochs_s_9, 
                epochs_t_8, epochs_t_7, epochs_s_6, epochs_s_5, epochs_s_4, epochs_s_3]

# %% NORMALIZATION

power_t_list_orig = power_t_list  
power_s_list_orig = power_s_list 

# BASELINE (None,None)
i = 0 
for i in range(len(power_t_list)):
    power_s_list[i].apply_baseline(baseline = (None, None), mode = 'zscore', verbose=None)
    i+= 1

i = 0 
for i in range(len(power_s_list)):
    power_t_list[i].apply_baseline(baseline = (None, None), mode = 'zscore', verbose=None)
    i+= 1

#CROPPING
i = 0 
for i in range(len(power_t_list)):
    power_t_list[i].crop(0,4)
    i+= 1

i = 0 
for i in range(len(power_s_list)):
    power_s_list[i].crop(0,4)
    i+= 1

#CREATING DATA LIST
i = 0 
for i in range(len(power_t_list)):
    power_data[i] = power_t_list[i].data
    i+= 1
    
i = 0 
for i in range(len(power_s_list)):
    power_data[i+26] = power_s_list[i].data
    i+= 1
    
i = 0 
for i in range(len(power_t_list)):
    power_t_data[i] = power_t_list[i].data
    i+= 1
    
i = 0 
for i in range(len(power_s_list)):
    power_s_data[i] = power_s_list[i].data
    i+= 1
    
# %% SAVING NORMALIZED .FIF OBJECT

#SAVING
i = 0 
for i in range(len(power_t_list)):
    power_t_list[i].save('S{}_power_t-normalized-tfr.h5'.format(i), overwrite = True)
    i+= 1

i = 0 
for i in range(len(power_s_list)):
    power_s_list[i].save('S{}_power_s-normalized-tfr.h5'.format(i), overwrite = True)
    i+= 1


# %% LOADING  NORMALIZED .FIF OBJECT

i = 0 
power_t_loaded_list = []
for i in range(25):
    power_tm = mne.time_frequency.read_tfrs('S{}_power_t-normalized-tfr.h5'.format(i))
    power_tn = power_tm
    power_t_loaded_list.append(power_tn)
    i+= 1 #NOW WE HAVE NORMALIZED TFR hidden by 2 lists [0] - to access another list and again [0] - to access the averageTFR

i = 0 
power_s_loaded_list = []
for i in range(25):
    power_sm = mne.time_frequency.read_tfrs('S{}_power_s-normalized-tfr.h5'.format(i))
    power_sn = power_sm
    power_s_loaded_list.append(power_sn)
    i+= 1 #NOW WE HAVE NORMALIZED TFR hidden by 2 lists [0] - to access another list and again [0] - to access the averageTFR
    

# %% CHECKING

#ORIGINAL
for i in range(len(power_t_list_orig)):
    power_t_list_orig[i].plot(combine='rms',title = 'subject T {}'.format(i))
    i+=1

for i in range(len(power_t_list_orig)):  
    power_s_list_orig[i].plot(combine='rms',title = 'subject S {}'.format(i))
    i+=1

#ORIGINAL
i = 0 
for i in range(len(power_s_list)):
    power_s_list[i].plot(combine='rms',title = 'subject S {}'.format(i))
    i+=1

i = 0 
for i in range(len(power_t_list)):
    #power_t_list[i].plot_joint(title = 'subject T {}'.format(i))
    power_t_list[i].plot(combine='rms', title = 'second T {}'.format(i))
    i+=1


#THIS GIVES ONE PLOT FOR ALL CHANNELS
i = 10
power_t_list[i].plot_joint(title = 'subject T {}'.format(i))
power_t_list[i].plot(combine='mean', title = 'second T {}'.format(i))


# %% AVERAGING
power_s_average = mne.grand_average(power_s_list, interpolate_bads=True, drop_bads=True)
power_t_average = mne.grand_average(power_t_list, interpolate_bads=True, drop_bads=True)
evoked_s_average = mne.grand_average(evoked_s_list, interpolate_bads=True, drop_bads=True)
evoked_t_average = mne.grand_average(evoked_t_list, interpolate_bads=True, drop_bads=True)

power_d = mne.grand_average(power_data, interpolate_bads=True, drop_bads=True)

#
evoked_s_average.plot_topo()
evoked_t_average.plot_topo()
power_s_average.plot_topo()
power_t_average.plot_topo()

#Diff
mne.viz.plot_compare_evokeds(dict(spatial=evoked_s_average, temporal=evoked_t_average),
                               legend='upper left', show_sensors='upper right')
evoked_diff = mne.combine_evoked([evoked_s_average, evoked_t_average], weights=[1, 1])
evoked_diff.plot_topo(color='r', legend=False)


