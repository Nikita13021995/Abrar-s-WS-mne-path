# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 08:51:05 2022

@author: Nikita O
"""
# %% LIBRARIES

import os
import os.path as op

import numpy as np
from numpy.random import randn
from scipy import stats as stats

import matplotlib.pyplot as plt

import mne
from mne.datasets import sample
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
from mne.viz import circular_layout
from mne_connectivity import spectral_connectivity_epochs
from mne_connectivity.viz import plot_connectivity_circle

import mne
from mne.epochs import equalize_epoch_counts
from mne.stats import (spatio_temporal_cluster_1samp_test,
                       summarize_clusters_stc)
from mne.minimum_norm import apply_inverse, read_inverse_operator
from mne.datasets import sample
from mne import read_source_estimate
from mne.minimum_norm import read_inverse_operator
import scipy
%matplotlib qt

import conpy
import sklearn
import h5io
import h5py
import nilearn

folder_with_files         = 'T:/to github/'                                     ###!!!CHANGABLE!!!
SDF                       = folder_with_files
data_path                 = folder_with_files + '/freesurfer'
trans_path                = folder_with_files

##### FOR CYCLES
index_array_2             = [3,6,7,9,11, 14, 18,19,20,21,23,25,26,27,28] 

# %% Average Source Space
os.chdir(folder_with_files)      
src_avg                 = mne.setup_source_space('fsaverage', spacing = 'oct6',  subjects_dir = data_path) 
mne.write_source_spaces('Sub_avg-oct6-src.fif', src_avg, overwrite = True)

# %% MORPHING

#####  AVERAGE SUBJECT
os.chdir(folder_with_files)     
src_surf_fs               = mne.read_source_spaces('Sub_avg-oct6-src.fif')

index = 3
for index in index_array_2:  
   SDRF                   = os.path.join(SDF, 'S{}_full_reconst_filt.fif'.format(index))
   raw_index              = mne.io.read_raw_fif(SDRF, allow_maxshield=False, preload=False,
                                         on_split_missing='raise', verbose=None)
 
   subject_src = mne.morph_source_spaces(src_from      = src_surf_fs, 
                                         subject_to    = 'Sub{}'.format(index), 
                                         subjects_dir  = data_path, 
                                         verbose=None)
   os.chdir(folder_with_files)      
   mne.write_source_spaces('Sub{}-mor-src.fif'.format(index), subject_src, overwrite = True)

# %% FORWARD MODELLING for morphed subject

index = 3
for index in index_array_2:
    i = index
    SDRF                   = os.path.join(SDF, 'S{}_full_reconst_filt.fif'.format(index))
    raw_index              = mne.io.read_raw_fif(SDRF, allow_maxshield=False, preload=False,
                                          on_split_missing='raise', verbose=None)
    os.chdir(folder_with_files) 
    src = mne.read_source_spaces('Sub{}-mor-src.fif'.format(index))
    
    ### TRANS FILE
    trans                  = os.path.join(trans_path, 'Subject{}-trans.fif'.format(index))
    
    verts                  = conpy.select_vertices_in_sensor_range( src, 
                                                                   dist=0.07, 
                                                                   info=raw_index.info, 
                                                                   trans = trans) 
    src_sub                = conpy.restrict_src_to_vertices( src, verts)

    os.chdir(folder_with_files) 
    bem = mne.read_bem_solution('Sub{}-dec-ind-bem-sol.fif'.format(index))
    
    fwd   = mne.make_forward_solution(
                                    raw_index.info,
                                    trans=trans,
                                    src=src_sub,
                                    bem=bem,
                                    meg=True,
                                    eeg=False,
                                    mindist= 0,
                                    n_jobs=6)                                     ###!!!CHANGABLE!!!
  
    os.chdir(folder_with_files) 
    mne.write_forward_solution('Sub{}-MOR-fwd.fif'.format(index), fwd, overwrite=True)
    
#### AVERAGE SUBJECT
os.chdir(folder_with_files)     
src_surf_fs            = mne.read_source_spaces('Sub_avg-oct6-src.fif')

trans                  = os.path.join(trans_path, 'fsaverage-trans.fif')
verts                  = conpy.select_vertices_in_sensor_range( src, 
                                                               dist=0.07, 
                                                               info=raw_index.info, 
                                                               trans = trans) 
src_sub                = conpy.restrict_src_to_vertices( src, verts)

os.chdir(folder_with_files)   
model                  = mne.make_bem_model(subject='fsaverage', ico=5,  #ICO 5 → 10240 downsampling
                          conductivity=(0.3,), 
                          subjects_dir=data_path)
bem                    = mne.make_bem_solution(model)
fwd                    = mne.make_forward_solution(
                                raw_index.info,
                                trans=trans,
                                src=src_sub,
                                bem=bem,
                                meg=True,
                                eeg=False,
                                mindist= 0,
                                n_jobs=8)

os.chdir(folder_with_files)   
mne.write_forward_solution('Sub_ave-fwd.fif'.format(index), fwd, overwrite=True)


# %% PREPARATION TO CONNECTIVITY

#### UPLOADER
fwd_ind                =  [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]
src_surf               =  [1,2,3,4,5,6,7,8,9,0,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29]

os.chdir(folder_with_files)   
src_surf_fs            = mne.read_source_spaces('Sub_avg-oct6-src.fif')
fwd_ave                = mne.read_forward_solution('Sub_ave-fwd.fif') 

index = 3
for index in index_array_2:  
    os.chdir(folder_with_files)   
    fwd_ind[index]     = mne.read_forward_solution('Sub{}-MOR-fwd.fif'.format(index)) 
    fwd_ind[index]     = conpy.forward_to_tangential( fwd_ind[index] )
    os.chdir(folder_with_files)   
    src_surf[index]    = mne.read_source_spaces('Sub{}-mor-src.fif'.format(index))

### LIST OF SUBJECT   
index                  = 3
for index in index_array_2[:]:    
    Subject            = 'Sub{}'.format(index)
    sub_list[index]    = Subject
    

### FINDING THE SHARED VERTICES
fwd_ind                = np.delete(fwd_ind, [0,1,2,4,5, 8,10,12,13,15,16,17,22,24,29])
src_surf               = np.delete(src_surf, [0,1,2,4,5, 8,10,12,13,15,16,17,22,24,29])

max_sensor_dist        = 0.07
fwd_ind[1]             = conpy.restrict_forward_to_sensor_range(fwd_ind[1], max_sensor_dist)

vert_inds              = conpy.select_shared_vertices(fwd_ind, fwd_ave, data_path)

for fwd, vert_ind, index in zip(fwd_ind, vert_inds, index_array_2):
    fwd_r = conpy.restrict_forward_to_vertices(fwd, vert_ind)
    os.chdir('L:/{}/'.format(folder))
    mne.write_forward_solution('Sub{}-oct6-commonvertices-surf-fwd.fif', fwd_r,
                               overwrite=True)
    

# %% CONNECTIVITY - PAIRS


index = 3
for index in index_array_2:  
    os.chdir('L:/SATURN/')
    fwd_ind[index]    = mne.read_forward_solution('Sub{}-oct6-commonvertices-surf-fwd.fif'.format(index)) 
   
min_pair_dist = 0.04
pairs = conpy.all_to_all_connectivity_pairs(fwd_ind[index] , min_dist=min_pair_dist)