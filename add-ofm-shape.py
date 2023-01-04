import onnx
import onnx.shape_inference

def find_node_by_input(nodes, input_name):
    for node in nodes:
        for input_ in node.input:
            if input_ == input_name:
                return node

def get_attr(node, attr_name):
    for attr in node.attribute:
        if attr.name != attr_name:
            continue
        return onnx.helper.get_attribute_value(attr)

    # Not found
    return None

def split2slice(model):
    new_nodes = []
    for node in model.graph.node:
        if node.op_type != 'Split':
            new_nodes.append(node)
            continue

        axis = get_attr(node, 'axis')
        split = get_attr(node, 'split')

        cur_start = 0
        cur_end = split[0]
        for idx, output in enumerate(node.output):
            output_node = find_node_by_input(model.graph.node, output)
            model.graph.initializer.extend([
                onnx.helper.make_tensor(f'{node.name}_start{idx}', onnx.TensorProto.INT64, [1], [cur_start]),
                onnx.helper.make_tensor(f'{node.name}_end{idx}', onnx.TensorProto.INT64, [1], [cur_end]),
                onnx.helper.make_tensor(f'{node.name}_axis{idx}', onnx.TensorProto.INT64, [1], [axis]),
            ])
            cur_start = cur_end
            if idx < len(node.output) - 1:
                cur_end += split[idx+1]
            new_node_inputs = [node.input[0], f'{node.name}_start{idx}', f'{node.name}_end{idx}', f'{node.name}_axis{idx}']
            new_node_output = f'{node.name}_output{idx}'
            new_nodes.append(onnx.helper.make_node('Slice', inputs=new_node_inputs, outputs=[new_node_output], name=f'{node.name}_slice{idx}'))
            replaced_inputs = [new_node_output if inp == output else inp for inp in output_node.input]
            del output_node.input[:]
            output_node.input.extend(replaced_inputs)

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

model = onnx.load_model('yolov5n.onnx')

split2slice(model)

model = onnx.shape_inference.infer_shapes(model)

onnx.save_model(model, 'yolov5n.onnx')
