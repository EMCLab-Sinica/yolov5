import onnx
import onnx.shape_inference

model = onnx.load_model('yolov5n.onnx')

model = onnx.shape_inference.infer_shapes(model)

onnx.save_model(model, 'yolov5n.onnx')
