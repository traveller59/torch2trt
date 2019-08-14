import numpy as np
import tensorrt as trt
import torch
from torch.nn import functional as F

from torch2trt.core import (current_context, has_trt_tensor, has_tvm_tensor,
                            register_node_handler)
from torch2trt.handlers.ops import _scale_or_elementwise
from torch2trt.utils import print_inputs

try:
    import tvm 
    from tvm.relay import expr as _expr
    from tvm.relay import op as _op
    from tvm import nd as _nd


except ImportError:
    pass


@register_node_handler("aten::relu_")
def aten_relu_(inputs, attributes, scope):
    ctx = current_context()
    net = current_context().network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        layer = net.add_activation(inputs[0], trt.ActivationType.RELU)
        output = layer.get_output(0)
        output.name = scope
        layer.name = scope
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        return [_op.nn.relu(inputs[0])]
    return [F.relu_(inputs[0])]


@register_node_handler("aten::relu")
def aten_relu(inputs, attributes, scope):
    ctx = current_context()
    net = current_context().network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        layer = net.add_activation(inputs[0], trt.ActivationType.RELU)
        output = layer.get_output(0)
        output.name = scope
        layer.name = scope
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        return [_op.nn.relu(inputs[0])]

    return [F.relu(inputs[0])]


@register_node_handler("aten::sigmoid")
def aten_sigmoid(inputs, attributes, scope):
    ctx = current_context()
    net = current_context().network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        layer = net.add_activation(inputs[0], trt.ActivationType.SIGMOID)
        output = layer.get_output(0)
        output.name = scope
        layer.name = scope
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        return [_op.sigmoid(inputs[0])]

    return [torch.sigmoid(inputs[0])]


@register_node_handler("aten::leaky_relu")
def aten_leaky_relu(inputs, attributes, scope):
    inp, leak = inputs[:2]
    ctx = current_context()
    net = current_context().network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        layer = net.add_activation(inp, trt.ActivationType.LEAKY_RELU)
        layer.alpha = leak
        output = layer.get_output(0)
        output.name = scope
        layer.name = scope
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        return [_op.nn.leaky_relu(inputs[0], leak)]

    return [F.leaky_relu(inp, leak)]


@register_node_handler("aten::leaky_relu_")
def aten_leaky_relu_(inputs, attributes, scope):
    inp, leak = inputs[:2]
    ctx = current_context()
    net = current_context().network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        layer = net.add_activation(inp, trt.ActivationType.LEAKY_RELU)
        layer.alpha = leak
        output = layer.get_output(0)
        output.name = scope
        layer.name = scope
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        return [_op.nn.leaky_relu(inputs[0], leak)]

    return [F.leaky_relu_(inp, leak)]


@register_node_handler("aten::tanh")
def aten_tanh(inputs, attributes, scope):
    inp = inputs[:1]
    ctx = current_context()
    net = current_context().network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        layer = net.add_activation(inp, trt.ActivationType.TANH)
        output = layer.get_output(0)
        output.name = scope
        layer.name = scope
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        return [_op.tanh(inputs[0])]

    return [F.tanh(inp)]


@register_node_handler("aten::elu")
def aten_elu(inputs, attributes, scope):
    inp, alpha = inputs[:2]
    ctx = current_context()
    net = current_context().network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        layer = net.add_activation(inputs[0], trt.ActivationType.ELU)
        layer.alpha = alpha
        output = layer.get_output(0)
        output.name = scope
        layer.name = scope
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        return [_op.nn.elu(inputs[0], alpha)]

    return [F.elu(inputs[0], alpha)]


@register_node_handler("aten::softsign")
def aten_softsign(inputs, attributes, scope):
    inp = inputs[0]
    ctx = current_context()
    net = current_context().network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        layer = net.add_activation(inputs[0], trt.ActivationType.SOFTSIGN)
        output = layer.get_output(0)
        output.name = scope
        layer.name = scope
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        raise NotImplementedError

    return [F.softsign(inputs[0])]


@register_node_handler("aten::softplus")
def aten_softplus(inputs, attributes, scope):
    inp, beta, thresh = inputs[:3]
    ctx = current_context()
    net = current_context().network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        layer = net.add_activation(inputs[0], trt.ActivationType.SOFTPLUS)
        layer.alpha = 1 / beta
        layer.beta = beta
        print("WARNING: tensorrt don't support threshold for softsign")
        output = layer.get_output(0)
        output.name = scope
        layer.name = scope
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        raise NotImplementedError

    return [F.softplus(inputs[0], beta, thresh)]


@register_node_handler("aten::hardtanh")
def aten_hardtanh(inputs, attributes, scope):
    inp, min_val, max_val = inputs[:3]
    ctx = current_context()
    net = current_context().network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        # use relu(x) - relu(x - 6) to implement relu6 (subset of hardtanh)
        # relu(x) - relu(x - 6) implementation is faster than hardsigmoid implementation
        assert min_val == 0, "only support relu6"
        layer = net.add_activation(inp, trt.ActivationType.RELU)
        output = layer.get_output(0)
        layer.name = scope + "/relu"
        tensor = np.full([1] * len(inp.shape), max_val, dtype=np.float32)
        trt_6 = ctx.network.add_constant([1] * len(inp.shape), tensor)
        layer = ctx.network.add_elementwise(output, trt_6.get_output(0), trt.ElementWiseOperation.MIN)
        output = layer.get_output(0)
        layer.name = scope + "/elem_min"
        output.name = scope + "/relu6"
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        raise NotImplementedError
    return [F.hardtanh(inp, min_val, max_val)]

@register_node_handler("aten::hardtanh_")
def aten_hardtanh_(inputs, attributes, scope):
    inp, min_val, max_val = inputs[:3]
    ctx = current_context()
    net = current_context().network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        # use relu(x) - relu(x - 6) to implement relu6 (subset of hardtanh)
        assert min_val == 0, "only support relu6"
        layer = net.add_activation(inp, trt.ActivationType.RELU)
        output = layer.get_output(0)
        layer.name = scope + "/relu"
        tensor = np.full([1] * len(inp.shape), max_val, dtype=np.float32)
        trt_6 = ctx.network.add_constant([1] * len(inp.shape), tensor)
        layer = ctx.network.add_elementwise(output, trt_6.get_output(0), trt.ElementWiseOperation.MIN)
        output = layer.get_output(0)
        layer.name = scope + "/elem_min"
        output.name = scope + "/relu6"
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        raise NotImplementedError
    return [F.hardtanh_(inp, min_val, max_val)]
