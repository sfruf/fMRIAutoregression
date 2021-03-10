'''
Calculate lag-1 Autoregression features from the Standardized Time Series matrix

'''

import numpy as np
from statsmodels.tsa.ar_model import AutoReg

def make_ar_array(ts_mat:np.ndarray) -> np.ndarray:
    '''

    Calculate a lag one Autoregressive model for each regional timecourse for each subject. 
    
    Assume time series is a (number of Regions, number of Timepoints, number of Subjects) array.
    
    '''

    num_regions=ts_mat.shape[0]
    num_subj=ts_mat.shape[2]

    autoreg_array=np.zeros((num_regions,num_subj))
    for reg_ind in range(num_regions):
        for sub_ind in range(num_subj):
            ar_model=AutoReg(ts_mat[reg_ind,:,sub_ind],lags=1,old_names=False).fit()
            autoreg_array[reg_ind,sub_ind]=ar_model.params[1] #ar_model has two params, a constant term (params[0]) and the coefficient for lag 1 (params [1])

    #Just in case something a little more sophisticated needs to be done in the future. 
    #The next step towards a more appropriate handling of the time series is to call 
    #ar_model=ar_select_order(data,maxlag=20).model.fit() #searches for the optimal model order up to 20 lags 
    #and then select the appropriate lag. The thing to watch out for here is that if lag 1 isn't optimal it won't be calculated in this method, so there would need to be a check that optimal order includes lag 1. To do that you can split the call into 
    #mod=ar_select_order(data,maxlag=20) 
    #ar_model=mod.model.fit()
    #and check 
    #mod.ar_lags to find the optimal lags for the model        
    return autoreg_array

if __name__=='__main__':
    data_dir='../../data/'
    ts_mat=np.load(f'{data_dir}interim/Standardized_Time_Series.npy')
    autoreg_array=make_ar_array(ts_mat)
    np.save(f'{data_dir}processed/Autoreg.npy',autoreg_array)