import tensorrt as trt
import torch

from torch2trt.core import (current_context, has_trt_tensor, has_tvm_tensor,
                            register_node_handler)
from torch2trt.utils import print_inputs
try:
    import tvm 
    from tvm.relay import expr as _expr
    from tvm.relay import op as _op
    from tvm import nd as _nd


except ImportError:
    pass



@register_node_handler("aten::cos")
def aten_cos(inputs, attributes, scope):
    inp = inputs[0]
    ctx = current_context()
    net = ctx.network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        layer = net.add_unary(inp, trt.UnaryOperation.COS)
        output = layer.get_output(0)
        output.name = scope
        layer.name = scope
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        raise NotImplementedError
    return [torch.cos(inp)]


@register_node_handler("aten::acos")
def aten_acos(inputs, attributes, scope):
    inp = inputs[0]
    ctx = current_context()
    net = ctx.network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        layer = net.add_unary(inp, trt.UnaryOperation.ACOS)
        output = layer.get_output(0)
        output.name = scope
        layer.name = scope
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        raise NotImplementedError

    return [torch.acos(inp)]


@register_node_handler("aten::cosh")
def aten_cosh(inputs, attributes, scope):
    inp = inputs[0]
    ctx = current_context()
    net = ctx.network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        layer = net.add_unary(inp, trt.UnaryOperation.COSH)
        output = layer.get_output(0)
        output.name = scope
        layer.name = scope
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        raise NotImplementedError

    return [torch.cosh(inp)]


@register_node_handler("aten::sin")
def aten_sin(inputs, attributes, scope):
    inp = inputs[0]
    ctx = current_context()
    net = ctx.network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        layer = net.add_unary(inp, trt.UnaryOperation.SIN)
        output = layer.get_output(0)
        output.name = scope
        layer.name = scope
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        raise NotImplementedError

    return [torch.sin(inp)]


@register_node_handler("aten::asin")
def aten_asin(inputs, attributes, scope):
    inp = inputs[0]
    ctx = current_context()
    net = ctx.network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        layer = net.add_unary(inp, trt.UnaryOperation.ASIN)
        output = layer.get_output(0)
        output.name = scope
        layer.name = scope
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        raise NotImplementedError

    return [torch.asin(inp)]


@register_node_handler("aten::sinh")
def aten_sinh(inputs, attributes, scope):
    inp = inputs[0]
    ctx = current_context()
    net = ctx.network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        layer = net.add_unary(inp, trt.UnaryOperation.SINH)
        output = layer.get_output(0)
        output.name = scope
        layer.name = scope
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        raise NotImplementedError

    return [torch.sinh(inp)]


@register_node_handler("aten::tan")
def aten_tan(inputs, attributes, scope):
    inp = inputs[0]
    ctx = current_context()
    net = ctx.network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        layer = net.add_unary(inp, trt.UnaryOperation.TAN)
        output = layer.get_output(0)
        output.name = scope
        layer.name = scope
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        raise NotImplementedError

    return [torch.tan(inp)]


@register_node_handler("aten::atan")
def aten_atan(inputs, attributes, scope):
    inp = inputs[0]
    ctx = current_context()
    net = ctx.network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        layer = net.add_unary(inp, trt.UnaryOperation.ATAN)
        output = layer.get_output(0)
        output.name = scope
        layer.name = scope
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        raise NotImplementedError

    return [torch.atan(inp)]


@register_node_handler("aten::abs")
def aten_abs(inputs, attributes, scope):
    inp = inputs[0]
    ctx = current_context()
    net = ctx.network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        layer = net.add_unary(inp, trt.UnaryOperation.ABS)
        output = layer.get_output(0)
        output.name = scope
        layer.name = scope
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        raise NotImplementedError

    return [torch.abs(inp)]


@register_node_handler("aten::floor")
def aten_floor(inputs, attributes, scope):
    inp = inputs[0]
    ctx = current_context()
    net = ctx.network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        layer = net.add_unary(inp, trt.UnaryOperation.FLOOR)
        output = layer.get_output(0)
        output.name = scope
        layer.name = scope
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        return [_op.floor(inp)]

    return [torch.floor(inp)]


@register_node_handler("aten::reciprocal")
def aten_reciprocal(inputs, attributes, scope):
    inp = inputs[0]
    ctx = current_context()
    net = ctx.network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        layer = net.add_unary(inp, trt.UnaryOperation.RECIP)
        output = layer.get_output(0)
        output.name = scope
        layer.name = scope
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        raise NotImplementedError

    return [torch.reciprocal(inp)]


@register_node_handler("aten::log")
def aten_log(inputs, attributes, scope):
    inp = inputs[0]
    ctx = current_context()
    net = ctx.network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        layer = net.add_unary(inp, trt.UnaryOperation.LOG)
        output = layer.get_output(0)
        output.name = scope
        layer.name = scope
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        return [_op.log(inp)]

    return [torch.log(inp)]


@register_node_handler("aten::ceil")
def aten_ceil(inputs, attributes, scope):
    inp = inputs[0]
    ctx = current_context()
    net = ctx.network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        layer = net.add_unary(inp, trt.UnaryOperation.CEIL)
        output = layer.get_output(0)
        output.name = scope
        layer.name = scope
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        return [_op.ceil(inp)]

    return [torch.ceil(inp)]


@register_node_handler("aten::sqrt")
def aten_sqrt(inputs, attributes, scope):
    inp = inputs[0]
    ctx = current_context()
    net = ctx.network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        layer = net.add_unary(inp, trt.UnaryOperation.SQRT)
        output = layer.get_output(0)
        output.name = scope
        layer.name = scope
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        return [_op.sqrt(inp)]

    return [torch.sqrt(inp)]


@register_node_handler("aten::exp")
def aten_exp(inputs, attributes, scope):
    inp = inputs[0]
    ctx = current_context()
    net = ctx.network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        layer = net.add_unary(inp, trt.UnaryOperation.EXP)
        output = layer.get_output(0)
        output.name = scope
        layer.name = scope
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        return [_op.exp(inp)]

    return [torch.exp(inp)]


@register_node_handler("aten::neg")
def aten_neg(inputs, attributes, scope):
    inp = inputs[0]
    ctx = current_context()
    net = ctx.network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        layer = net.add_unary(inp, trt.UnaryOperation.NEG)
        output = layer.get_output(0)
        output.name = scope
        layer.name = scope
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        return [_op.negative(inp)]

    return [torch.neg(inp)]