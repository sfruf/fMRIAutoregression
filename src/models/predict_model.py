def predict_model_from_name(model_name:str,x_predict):
    ''' Make a prediction based on the name of the model'''

    import pickle
    import os

    workspace=os.environ['workspaceFolder']

    #check model name is a pickle file
    if model_name[-4:]!=".sav": 
        model_name+=".sav"

    #check model name is a path
    if model_name[0]!="/" or model_name[0]!=".":
        model_path=f"{workspace}models/{model_name}"
    else: 
        model_path=model_name


    with open(model_path,"rb") as f:
        model=pickle.load(f)

    prediction=model.predict(x_predict)

    return prediction