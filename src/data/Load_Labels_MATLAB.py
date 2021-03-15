'''
Load subject labels (depressed,anxious,dep/anx, or control) from .mat files, then save in python format
'''
from scipy.io import loadmat
import numpy as np

if __name__=='__main__':
    data_dir='../../data/'
    label_dict=loadmat(f'{data_dir}raw/All_inds.mat')
    labels=label_dict['all_inds'].flatten() # Labels are 1 control, 2 anxious, 3 depressed and comorbid anxious/depressed
    np.save(f'{data_dir}processed/Class_Labels.npy',labels)
    two_class_labels=(labels>1).astype(int) # Two_Class_Labels are 1 depressed and/or anxious, 0 control
    np.save(f'{data_dir}processed/Two_Class_Labels.npy',two_class_labels)
