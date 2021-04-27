# Model Notes

I've recorded the potential hyperparameter settings here in case I need to go back and retrain the original models.

## Models with version 0.0 were trained with possible hyperparameter values

    num_components=[4,6,8,10,12]
    svc_kernels=['linear','rbf']
    svc_reg=[.1,1,10,100,1000] 
    depths=[*range(1,8)]

## Models with version 0.1 were trained with possible hyperparameter values

The space of hyperparameters was reduced to allow for the PCA step to be included. 

    num_components=[4,6,8]
    svc_kernels=['linear','rbf']
    svc_reg=[.1,1,10,100] 
    depths=[*range(1,8,2)]
