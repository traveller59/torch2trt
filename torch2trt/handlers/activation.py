import tensorrt as trt
import torch
from torch.nn import functional as F
from torch2trt.core import (current_network, has_trt_tensor,
                            register_node_handler)
from torch2trt.utils import print_inputs


@register_node_handler("aten::relu_")
def aten_relu_(inputs, attributes, scope):
    net = current_network()
    if net is not None and has_trt_tensor(inputs):
        layer = net.add_activation(inputs[0], trt.ActivationType.RELU)
        output = layer.get_output(0)
        output.name = scope
        layer.name = scope
        return [output]
    return [F.relu_(inputs[0])]


@register_node_handler("aten::relu")
def aten_relu(inputs, attributes, scope):
    net = current_network()
    if net is not None and has_trt_tensor(inputs):
        layer = net.add_activation(inputs[0], trt.ActivationType.RELU)
        output = layer.get_output(0)
        output.name = scope
        layer.name = scope
        return [output]
    return [F.relu(inputs[0])]


@register_node_handler("aten::sigmoid")
def aten_sigmoid(inputs, attributes, scope):
    net = current_network()
    if net is not None and has_trt_tensor(inputs):
        layer = net.add_activation(inputs[0], trt.ActivationType.SIGMOID)
        output = layer.get_output(0)
        output.name = scope
        layer.name = scope
        return [output]
    return [torch.sigmoid(inputs[0])]


@register_node_handler("aten::leaky_relu")
def aten_leaky_relu(inputs, attributes, scope):
    inp, leak = inputs[:2]
    net = current_network()
    if net is not None and has_trt_tensor(inputs):
        layer = net.add_activation(inp, trt.ActivationType.LEAKY_RELU)
        layer.alpha = leak
        output = layer.get_output(0)
        output.name = scope
        layer.name = scope
        return [output]
    return [F.leaky_relu(inp, leak)]


@register_node_handler("aten::tanh")
def aten_tanh(inputs, attributes, scope):
    inp = inputs[:1]
    net = current_network()
    if net is not None and has_trt_tensor(inputs):
        layer = net.add_activation(inp, trt.ActivationType.TANH)
        output = layer.get_output(0)
        output.name = scope
        layer.name = scope
        return [output]
    return [F.tanh(inp)]


@register_node_handler("aten::elu")
def aten_elu(inputs, attributes, scope):
    inp, alpha = inputs[:2]
    net = current_network()
    if net is not None and has_trt_tensor(inputs):
        layer = net.add_activation(inputs[0], trt.ActivationType.ELU)
        layer.alpha = alpha
        output = layer.get_output(0)
        output.name = scope
        layer.name = scope
        return [output]
    return [F.elu(inputs[0], alpha)]


@register_node_handler("aten::softsign")
def aten_softsign(inputs, attributes, scope):
    inp = inputs[0]
    net = current_network()
    if net is not None and has_trt_tensor(inputs):
        layer = net.add_activation(inputs[0], trt.ActivationType.SOFTSIGN)
        output = layer.get_output(0)
        output.name = scope
        layer.name = scope
        return [output]
    return [F.softsign(inputs[0])]


@register_node_handler("aten::softplus")
def aten_softplus(inputs, attributes, scope):
    inp, beta, thresh = inputs[:3]
    net = current_network()
    if net is not None and has_trt_tensor(inputs):
        layer = net.add_activation(inputs[0], trt.ActivationType.SOFTPLUS)
        layer.alpha = 1 / beta
        layer.beta = beta
        print("WARNING: tensorrt don't support threshold for softsign")
        output = layer.get_output(0)
        output.name = scope
        layer.name = scope
        return [output]
    return [F.softplus(inputs[0], beta, thresh)]
