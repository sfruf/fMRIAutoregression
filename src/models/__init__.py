def jackknife_variance(X,y,model):
    ''' Returns an estimate of the variance of the fit of classification model on (X,y) using jackknife resampling
    '''
    import numpy as np
    S=list()
    n=len(y)
    for ind in range(0,n):
        Samples=np.delete(X,ind,axis=0)
        Labels=np.delete(y,ind,axis=0)
        S.append(model.score(Samples,Labels))
    return (n-1)*np.var(S)

