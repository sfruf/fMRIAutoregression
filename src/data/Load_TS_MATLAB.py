'''
Load timeseries from .mat files, standardize, and then save in python format
'''

import numpy as np
from scipy.io import loadmat

if __name__=='__main__':

    data_dir='../../data/'
    mat_contents=loadmat(f'{data_dir}raw/Glaser_ts.mat') # load .mat file with timeseries in the Glaser Parcellation

    Time_Series_Matrix=mat_contents['ts_mat']
    np.save(f'{data_dir}raw/Time_Series.npy',Time_Series_Matrix)

    #standardize timeseries data
    num_Subj=Time_Series_Matrix.shape[2] 
    num_Regions=Time_Series_Matrix.shape[0]
    num_Timepoints=Time_Series_Matrix.shape[1]
    scan_length=num_Timepoints//4

    Time_Series_Matrix_standard=np.zeros(Time_Series_Matrix.shape)
    for reg_ind in range(num_Regions):
        for scan_ind in range(0,num_Timepoints,scan_length):
            for sub_ind in range(num_Subj):
                scan_ts=Time_Series_Matrix[reg_ind,scan_ind:scan_ind+scan_length,sub_ind]
                Time_Series_Matrix_standard[reg_ind,scan_ind:scan_ind+scan_length,sub_ind]=(scan_ts-np.mean(scan_ts))/np.std(scan_ts)

    np.save(f'{data_dir}interim/Standardized_Time_Series.npy',Time_Series_Matrix_standard)
