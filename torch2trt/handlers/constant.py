import tensorrt as trt
import torch

from torch2trt.core import (current_network, has_trt_tensor,
                            register_node_handler)
from torch2trt.utils import print_inputs


@register_node_handler("prim::Constant")
def prim_constant(inputs, attributes, scope):
    if "value" not in attributes:
        return [None]
    return [attributes["value"]]


@register_node_handler("prim::ListConstruct")
def prim_list_construct(inputs, attributes, scope):
    return [list(inputs)]


@register_node_handler("prim::TupleConstruct")
def prim_tuple_construct(inputs, attributes, scope):
    return [tuple(inputs)]


@register_node_handler("aten::size")
def aten_size(inputs, attributes, scope):
    axis = inputs[1]
    net = current_network()
    if net is not None and has_trt_tensor(inputs):
        # trt tensor shape don't include batch axis
        if axis == 0:
            return [-1]
        else:
            return [inputs[0].shape[inputs[1] - 1]]
    return [inputs[0].shape[inputs[1]]]


@register_node_handler("prim::NumToTensor")
def prim_num_to_tensor(inputs, attributes, scope):
    return [inputs[0]]


@register_node_handler("prim::Int")
def prim_int(inputs, attributes, scope):
    return [int(inputs[0])]


@register_node_handler("aten::to")
def aten_to(inputs, attributes, scope):
    inp, dst = inputs[:2]
    net = current_network()
    if net is not None and has_trt_tensor(inputs):
        raise NotImplementedError
    return [inp.to(dst)]


@register_node_handler("aten::detach")
def aten_detach(inputs, attributes, scope):
    inp = inputs[0]
    net = current_network()
    if net is not None and has_trt_tensor(inputs):
        raise NotImplementedError
    return [inp.detach()]


@register_node_handler("aten::t")
def aten_t(inputs, attributes, scope):
    inp = inputs[0]
    # weights in nn.Linear use this.
    assert isinstance(inp, torch.Tensor), "don't support this in tensorrt"
    return [inputs[0].t()]


@register_node_handler("prim::ListUnpack")
def prim_list_unpack(inputs, attributes, scope):
    inp = inputs[0]
    return [*inputs[0]]
