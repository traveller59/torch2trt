import tensorrt as trt
import torch

from torch2trt.core import (current_context, has_trt_tensor,
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

@register_node_handler("prim::NumToTensor")
def prim_num_to_tensor(inputs, attributes, scope):
    return [inputs[0]]


@register_node_handler("prim::Int")
def prim_int(inputs, attributes, scope):
    return [int(inputs[0])]

@register_node_handler("aten::Int")
def aten_int(inputs, attributes, scope):
    return [int(inputs[0])]

@register_node_handler("aten::to")
def aten_to(inputs, attributes, scope):
    inp, dst = inputs[:2]
    net = current_context().network
    if net is not None and has_trt_tensor(inputs):
        raise NotImplementedError
    res = inp.to(dst)
    if hasattr(inp, "__torch2trt_weight_name"):
        res.__torch2trt_weight_name = inp.__torch2trt_weight_name
    return [res]


@register_node_handler("aten::detach")
def aten_detach(inputs, attributes, scope):
    inp = inputs[0]
    net = current_context().network
    if net is not None and has_trt_tensor(inputs):
        raise NotImplementedError
    return [inp.detach()]


@register_node_handler("aten::t")
def aten_t(inputs, attributes, scope):
    inp = inputs[0]
    # weights in nn.Linear use this.
    assert isinstance(inp, torch.Tensor), "don't support this in tensorrt"
    res = inputs[0].t()
    if hasattr(inp, "__torch2trt_weight_name"):
        res.__torch2trt_weight_name = inp.__torch2trt_weight_name
    return [res]


@register_node_handler("prim::ListUnpack")
def prim_list_unpack(inputs, attributes, scope):
    return [*inputs[0]]

@register_node_handler("prim::GetAttr")
def prim_get_attr(inputs, attributes, scope):
    attr_name = attributes["name"]
    attr = getattr(inputs[0], attr_name)
    ctx = current_context()
    if isinstance(attr, torch.Tensor):
        attr.__torch2trt_weight_name = scope
        ctx.torch_weight_nodes_dict[scope] = attr
    return [attr]
