'''
Load network features from .mat files, then save in python format. 

There are three network featuers of interest, the positive nodal strength, the negative nodal strength, and the signed clustering coefficient.
These were calculated using functions from the Brain Connectivity Toolbox in Matlab: 
Network Measures/strength_und_sign.m for the strengths 
Network Measures/clustering_coef_wu_sign.m method $3$ for the clustering coefficient.
'''
from scipy.io import loadmat
import numpy as np

if __name__=='__main__':
    network_mat=loadmat('../../data/interim/Network_Measures.mat') # load .mat file with timeseries in the Glaser Parcellation
    data_path='../../data/processed/'
    pos_str=network_mat['pos']
    np.save(f'{data_path}Pos.npy',pos_str)

    neg_str=network_mat['neg']
    np.save(f'{data_path}Neg.npy',neg_str)

    clus_co=network_mat['cc']
    np.save(f'{data_path}Clus.npy',clus_co)

