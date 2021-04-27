
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


def binary_pca_plot(x_plot,y_plot,label_string,pca_kernel,pca_gamma,save_flag=False):
    ''' Transforms x_plot to two dimensions with kernel PCA and makes a scatter plot titled with label_string, colored by binary labels in y_plot. '''
    from sklearn.decomposition import KernelPCA
    import matplotlib as plt
    
    if x_plot.shape[0]>y_plot.shape[0]: 
        x_pca=KernelPCA(n_components=2,kernel=pca_kernel.lower(),gamma=pca_gamma).fit_transform(x_plot.transpose())
    else: 
        x_pca=KernelPCA(n_components=2,kernel=pca_kernel.lower(),gamma=pca_gamma).fit_transform(x_plot)

    plt.plot(x_pca[y_plot==1,0],x_pca[y_plot==1,1],'b+')
    plt.plot(x_pca[y_plot==0,0],x_pca[y_plot==0,1],'g+')

    plt.legend(labels=['Seizure','No Seizure'])
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.title(f'PCA plot of {label_string} with {pca_kernel} kernel')

    plt.show()

    if save_flag:
        plt.savefig(f"PCA {label_string} {pca_kernel} {str(pca_gamma)}.jpeg")