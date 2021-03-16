import onnxruntime as rt

sess=rt.InferenceSession("mdl_trained.onnx")
input_name=sess.get_inputs()[0].name
label_name=sess.get_outputs()[0].name

pred_onx=sess.run([label_name],{input_name:X_test.astype(numpy.float32)})[0]