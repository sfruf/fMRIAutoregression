def make_pipe(pipe_name:str):
    '''
    Makes a pipeline object based on the name. Looks for component_component.  

    '''
    from sklearn.svm import SVC
    from sklearn.pipeline import Pipeline
    from xgboost import XGBClassifier
    from sklearn.feature_selection import SelectKBest, mutual_info_classif
    from sklearn.decomposition import PCA

    pipe_parts=pipe_name.split('_')
    pipe_list=list()

    svc=SVC()
    xgb=XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    pca=PCA()
    mi=SelectKBest(mutual_info_classif)

    for part in pipe_parts:
        if part.lower() == 'mi':
            pipe_list.append(('MI',mi))
        elif part.lower() == 'svc':
            pipe_list.append(('SVC',svc))
        elif part.lower() == 'pca':
            pipe_list.append(('PCA',pca))
        elif part.lower()=='xgb':
            pipe_list.append(('XGB',xgb))

    try:
        pipeline=Pipeline(pipe_list)
    except TypeError:
        print('Pipeline is probably in the wrong order')
        pipeline="No Pipeline"

    return pipeline

def make_params(pipe_name):
    '''
    Makes a param grid based on the name. Looks for component_component.  
    '''

    pipe_parts=pipe_name.split('_')
    param_grid={}

    num_components=[4,6,8,10,12]
    svc_kernels=['linear','rbf']
    svc_reg=[.1,1,10,100, 1000] 
    depths=[*range(1,8)]

    params_svc = {'SVC__kernel': svc_kernels,'SVC__C': svc_reg}

    params_xgb = {'XGB__objective': ['binary:logistic'],'XGB__max_depth':depths }

    params_pca = {'pca__n_components':num_components}

    params_mi = {'MI__k':num_components}

    for part in pipe_parts:
        temp_dict=None
        if part.lower() == 'mi':
            temp_dict=params_mi
        elif part.lower() == 'svc':
            temp_dict=params_svc
        elif part.lower() == 'pca':
            temp_dict=params_pca
        elif part.lower()=='xgb':
            temp_dict=params_xgb
        if temp_dict:
            for key, value in temp_dict.items():
                param_grid[key]=value

    return param_grid

def multi_subset_pipeline(X,y,CV,subsets:dict,pipelines:list,save_flag): 
    '''
    Train the classifiers described by pipelines, using subsets of features described by subsets. 

    Work in Progress:
    I've copied all my code from autoregression over. I need to split it into things that are for evaluation 
    and things that should go into predicitons 
    '''
    scoring = {'Acc': 'accuracy', 'Bal_Acc': 'balanced_accuracy'} 

    for key,val in subsets.items():
        X_sub=X[:,val]
        for pipeline_name in pipelines:
            pipeline=make_pipe(pipeline_name)
            params=make_params(pipeline_name)
            search=GridSearchCV(estimator=pipe,param_grid=param,scoring=scoring,refit='Bal_Acc',cv=CV)
            search.fit(X_train_sc, y_train)
            All_feat_test.append(search.score(X_test_sc, y_test))
            All_feat_test_jack.append(jackknife_variance(X_test_sc,y_test,search))
            save_name=f'{workspaceFolder}/models/{key}/{pipeline_name}'

        print(f'For all features')
        print(f'Best score on training data is {search.best_score_}')
        print(f'Using a classifier with the following parameters {search.best_estimator_} \n')
        print(f'On the test set the classifier has an accuracy of {All_feat_test[-1]} with variance {All_feat_test_jack[-1]} and the following report \n')
        print(classification_report(y_test,search.predict(X_test_sc)))
        if save_flag:
            from skl2onnx import to_onnx
            onx=to_onnx(mdl,X_train[:,1].astype(numpy.float32))
            with open("mdl_trained.onnx","wb") as f:
                f.write(onx.SerializeToString())
    # program flow -> enter X, y, dict of {"subset_name":indices} to split the subset of features used to predict the model
    # [pipes] a list of pipeline names, CV object

#save models in format pipe_subset_folds.onnx
# return accuracy={"pipes_subset":train accuracy}
    return pipelines

#def multi_pipe(X,y,CV,pipes):
#
#    return trained_pipe


#

#onx=to_onnx(mdl,X_train[:,1].astype(numpy.float32))

#with open("mdl_trained.onnx","wb") as f:
#    f.write(onx.SerializeToString())