# program flow -> enter X_train, y, dict of {"subset_name":indices} to split the subset of features used to predict the model
# [pipes] a list of pipeline names, CV object

#save models in format pipe_subset_folds.onnx
# return accuracy={"pipes_subset":train accuracy}


from skl2onnx import to_onnx

onx=to_onnx(mdl,X_train[:,1].astype(numpy.float32))

with open("mdl_trained.onnx","wb") as f:
    f.write(onx.SerializeToString())