
def load_features(data_dir:str):
    '''
    Helper function to load all the features used in analysis

    class_labels,two_class_labels,pos_str,neg_str,clus_co,ar_array,num_regions,num_subjs=load_features(data_dir)

    Loads network and autoregressive features, labels, and calculates some useful shape parameters

    '''
    
    import numpy as np

    if data_dir[-1] != '/': #check path format
        data_dir.append('/')

    class_labels=np.load(f'{data_dir}processed/Class_Labels.npy')
    two_class_labels=np.load(f'{data_dir}processed/Two_Class_Labels.npy')
    pos_str=np.load(f'{data_dir}processed/Pos.npy')
    neg_str=np.load(f'{data_dir}processed/Neg.npy')
    clus_co=np.load(f'{data_dir}processed/Clus.npy')
    ar_array=np.load(f'{data_dir}processed/Autoreg.npy')

    num_regions=ar_array.shape[0]
    num_subjs=ar_array.shape[1]

    return class_labels,two_class_labels,pos_str,neg_str,clus_co,ar_array,num_regions,num_subjs


