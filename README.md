# torch2trt: PyTorch Module Instance to TensorRT

## Install

1. install pytorch 1.1+

2. download TensorRT 5.1.2.2+, install tensorrt python package, add TensorRT libraries to LD_LIBRARY_PATH.

3. clone this project, run ```python setup.py install```

## Usage

### Basic Example

#### Run in TensorRT mode

```Python
import torch
import torchvision
import tensorrt as trt
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

net = torchvision.models.inception_v3(pretrained=True).eval()
inputs = torch.rand(1, 3, 299, 299)

with trt.Builder(TRT_LOGGER) as builder, builder.create_network() as network:
    builder.max_workspace_size = 1 << 30
    # you can use either trt.ITensor or torch.Tensor as inputs.
    with torch2trt.core.trt_network(trt_net):
        trace, graph_pth = torch2trt.core.torch2trt(net, inputs, input_names=["image"], verbose=True, param_exclude=".*AuxLogits.*")
    results = graph_pth.get_resolved_outputs()
    print(results)
    output_tensor = results[0][0]
    output_tensor.name = "output"
    # you can convert another module:
    # plugin = net.add_plugin_v2(...) # add unsupported operator
    # output_tensor = plugin.get_output(0)
    """
    with torch2trt.core.trt_network(trt_net):
        trace, graph_pth = torch2trt.core.torch2trt(other_net, output_tensor, input_names=["other_input"], verbose=True, param_exclude=".*AuxLogits.*")
    results = graph_pth.get_resolved_outputs()
    output_tensor = results[0][0]
    output_tensor.name = "output2"
    """
    # you can add custom post process plugin here...
    trt_net.mark_output(tensor=output_tensor)
    engine = builder.build_cuda_engine(network)
    engine_bin = engine.serialize()
```

* Inputs and Outputs of TensorRT network

Inputs is inputs of net.forward, Outputs is outputs of net.forward.

* ```get_resolved_outputs```

return list of output for every output node.

* ```param_exclude```

torch2trt can't convert module with unused weights and buffers. if your module contains them, you need to use regex string to filter them.

#### Run in pytorch debug mode

```
import common # from tensorrt samples
import torchvision
import tensorrt as trt

net = torchvision.models.inception_v3(pretrained=True).eval()
inputs = torch.rand(1, 3, 224, 224)
with torch2trt.core.torch_network():
    trace, graph_pth = torch2trt.core.torch2trt(net, inputs, verbose=True, param_exclude=".*AuxLogits.*")

results = torch2trt.core.debug_call_graph(graph_pth, inputs)
```

### Add new handler

You can add handlers for missing nodes or tensorrt custom plugin.

```
@register_node_handler("aten::sum")
def aten_sum(inputs, attributes):
    inp, axes, keepdim = inputs
    net = current_network()
    if net is not None and has_trt_tensor(inputs):
        axis_trt = _axes_to_trt_axis(axes, len(inp.shape))
        layer = net.add_reduce(inp, trt.ReduceOperation.SUM, axis_trt,
                               bool(keepdim))
        output = layer.get_output(0)
        return [output]
    return [inp.sum(tuple(axes), keepdim=bool(keepdim))]
```

1. figure out the input format, you can check this [page](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/native_functions.yaml) as a reference.

2. use ```current_network``` to get current tensorrt INetworkDefinition instance. If None, the pytorch debug mode is enabled, you should implement pytorch code in this node handler for debugging.

3. use ```has_trt_tensor(inputs)``` to ensure inputs contains trt.ITensor. If there is no trt.ITensor, this node is a constant node and should be evaluated in pytorch mode.

4. return list of output tensors. all nodes MUST return list or tuple to handle node with multiple outputs.

5. for unsupported operators, write custom tensorrt plugin to handle this, then use net.add_plugin_v2 to add plugin in handler. you should also write custom torch jit operator for debugging. if you don't want to write jit operator, you can call torch2trt with several sub-module and connect them by tensorrt layers.

## Tips to write TensorRT-compatible modules

* all operations MUST use a static batch size.

* don't use any assign operation. TensorRT don't support assign/scatter.

* avoid to use any tensor create functions such as torch.zeros in forward code.

* write fused custom c++ jit operators for unsupported operations, also need to write a corresponding TensorRT plugin.

* don't add unused modules with weights to nn.Module. if you can't modify module code, use ```param_exclude``` in torch2trt to remove them.

* if your custom module have weights, you MUST use name contains "weight" or "bias", otherwise these weights will be filtered and cause error.