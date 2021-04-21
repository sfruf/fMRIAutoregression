from src.models import jackknife_variance


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


def train_multi_subset_pipeline(x,y,cv,subsets:dict,pipelines:list,save_flag): 
    '''
    Train the classifiers described by pipelines, using subsets of features described by subsets. 
    '''

    from sklearn.model_selection import GridSearchCV
    import numpy as np
    import os
    workspace=os.environ['workspaceFolder']

    scoring = {'Acc': 'accuracy', 'Bal_Acc': 'balanced_accuracy'} 
    score=list()
    estimator=list()
    fit_models=list()
    for key,val in subsets.items():
        x_sub=x[:,val] 
        for pipeline_name in pipelines:
            pipeline=make_pipe(pipeline_name)
            params=make_params(pipeline_name)
            search=GridSearchCV(estimator=pipeline,param_grid=params,scoring=scoring,refit='Bal_Acc',cv=cv)
            search.fit(x_sub, y)
            fit_models.append(search)
            score.append(search.best_score_)
            estimator.append(search.best_estimator_)
            save_name=f'{workspace}models/{key}_{pipeline_name}.sav'
        if save_flag:
            from pickle import dump

            with open(save_name,"wb") as f:
                dump(search,f)
    return fit_models,score,estimator




def score_on_test(x,y,model):
    ''' 
    Calculates model performance on test set using built in score method and jackknife resampling. 
    '''
    if isinstance(model,list):
        score=list()
        variance=list()
        for mod in model:
            score.append(mod.score(x,y))
            variance.append(jackknife_variance(x,y,mod))
    else:
        score=model.score(x,y)
        variance=jackknife_variance(x,y,model)

    return score,variance

def score_on_test_subset(x,y,subsets:dict,models):
    '''
    Calculates model performance on test set across multiple subsets of data
    '''
    
    if len(subsets)!= len(models):
        print("Model number doesn't match the number of data subsets")

    for ind,(key,val) in enumerate(subsets.items()):
        x_sub=x[:,val]
        for mod in models[ind]:
            score,variance=score_on_test(x_sub,y,mod)

    # Next steps: 
        
#    score.append(search.score(X_test_sc, y_test))
#    variance.append(jackknife_variance(X_test_sc,y_test,search))
#    print(f'For all features')
#    print(f'On the test set the classifier has an accuracy of {All_feat_test[-1]} with variance {All_feat_test_jack[-1]} and the following report \n')
#    print(classification_report(y_test,search.predict(X_test_sc)))
