import numpy as np
import tensorrt as trt
import torch
from torch.nn import functional as F

from torch2trt.core import (current_context, has_trt_tensor, has_tvm_tensor,
                            register_node_handler)
from torch2trt.utils import print_inputs

try:
    import tvm
    from torch2trt.utils import infer_shape, infer_dtype
    from tvm.relay import expr as _expr
    from tvm.relay import op as _op
    from tvm import nd as _nd

except ImportError:
    pass


@register_node_handler("aten::size")
def aten_size(inputs, attributes, scope):
    axis = inputs[1]
    ctx = current_context()
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        # trt tensor shape don't include batch axis
        if axis == 0:
            return [-1
                    ]  # can't be None because prim::Int may take this result.
        else:
            return [inputs[0].shape[inputs[1] - 1]]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        inp_shape = infer_shape(inputs[0])
        return [inp_shape[axis]]
    return [inputs[0].shape[inputs[1]]]

@register_node_handler("aten::view")
def aten_view(inputs, attributes, scope):
    assert len(inputs) == 2
    ctx = current_context()
    net = current_context().network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        shape = inputs[1][1:]
        # trt tensor shape don't include batch axis
        # TODO add batch size check
        if len(shape) == 1:
            shape += [1, 1]
        # elif len(shape) == 2:
        #     shape += [1]
        layer = net.add_shuffle(inputs[0])
        layer.reshape_dims = shape
        output = layer.get_output(0)
        layer.name = scope
        output.name = scope
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        return [_op.reshape(inputs[0], newshape=inputs[1])]
    return [inputs[0].view(*inputs[1])]

@register_node_handler("aten::clone")
def aten_clone(inputs, attributes, scope):
    ctx = current_context()
    net = current_context().network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        layer = net.add_identity(inputs[0])
        output = layer.get_output(0)
        layer.name = scope
        output.name = scope
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        raise NotImplementedError
    return [inputs[0].clone()]

@register_node_handler("aten::flatten")
def aten_flatten(inputs, attributes, scope):
    inp, start_dim, end_dim = inputs[:3]
    ctx = current_context()
    net = current_context().network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        assert len(inp.shape) == 3
        assert start_dim == 1 and (end_dim == -1 or end_dim == len(inp.shape))
        new_shape = [int(np.prod(list(inp.shape))), 1, 1]
        layer = net.add_shuffle(inputs[0])
        layer.reshape_dims = new_shape
        output = layer.get_output(0)
        layer.name = scope
        output.name = scope
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        assert len(infer_shape(inp)) == 4
        assert start_dim == 1 and (end_dim == -1 or end_dim == len(infer_shape(inp)) - 1)
        return [_op.nn.batch_flatten(inputs[0])]
    return [torch.flatten(*inputs)]

@register_node_handler("aten::reshape")
def aten_reshape(inputs, attributes, scope):
    return aten_view(inputs, attributes, scope)

@register_node_handler("aten::_convolution")
def aten_convolution(inputs, attributes, scope):
    inp, weight, bias = inputs[:3]
    stride, pad, dilation = inputs[3:6]
    transposed, output_padding, groups = inputs[6:9]
    ctx = current_context()
    net = ctx.network
    if transposed:
        I, O_groups, *ksize = weight.shape
        O = O_groups * groups
    else:
        O, I_groups, *ksize = weight.shape
        I = I_groups * groups
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        assert all([e == 0 for e in output_padding
                    ]), "tensor rt don't support out padding"
        ndim = len(ksize)
        if ndim == 1:
            print("WARNING: consider write conv2d because trt don't support conv2d, we need to change input shape (and output shape) and may cause error in following layers.")
            ksize = [ksize[0], 1]
            stride = [stride[0], 1]
            pad = [pad[0], 0]
            dilation = [dilation[0], 1]
        if len(inputs[0].shape) == 2:
            inputs[0] = _trt_reshape(net, inputs[0], [*inputs[0].shape, 1], scope + "/conv1d_reshape")

        assert ndim <= 2, "tensorrt only support 1d/2d conv"
        # trt weight format: GKCRS: [num_groups, O_groups, I, H, W]
        weight = weight.detach().cpu().numpy()
        if bias is not None:
            trt_bias = bias.detach().cpu().numpy()
        else:
            trt_bias = trt.Weights()
        if transposed:
            layer = net.add_deconvolution(inputs[0], O, tuple(ksize), weight,
                                          trt_bias)
        else:
            layer = net.add_convolution(inputs[0], O, tuple(ksize), weight,
                                        trt_bias)
            layer.dilation = tuple(dilation)
        layer.stride = tuple(stride)
        layer.padding = tuple(pad)
        layer.num_groups = groups
        output = layer.get_output(0)
        output.name = scope
        layer.name = scope
        ctx.refit_weight_dict[layer.name] = {
            "type": "Convolution",
            "weight": inputs[1].__torch2trt_weight_name,
        }
        if bias is not None:
            ctx.refit_weight_dict[layer.
                                  name]["bias"] = bias.__torch2trt_weight_name
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        weight = weight.detach().cpu().numpy()
        weight_t = _expr.var(
            scope + "/weight", shape=weight.shape, dtype="float32")
        ctx.tvm_weight_dict[weight_t] = weight
        ctx.refit_weight_dict[weight_t.name_hint] = inputs[1].__torch2trt_weight_name
        if bias is not None:
            bias = bias.detach().cpu().numpy()
            bias_t = _expr.var(
                scope + "/bias", shape=bias.shape, dtype="float32")
            ctx.tvm_weight_dict[bias_t] = bias
            ctx.refit_weight_dict[bias_t.name_hint] = bias.__torch2trt_weight_name
        new_attrs = {}
        new_attrs["channels"] = O
        new_attrs["kernel_size"] = ksize
        new_attrs["strides"] = stride
        new_attrs["padding"] = pad
        new_attrs["dilation"] = dilation
        new_attrs["groups"] = groups
        new_attrs["data_layout"] = "NCHW"
        new_attrs["kernel_layout"] = "OIHW"
        use_bias = bias is not None
        if transposed:
            new_attrs["output_padding"] = output_padding
            res = _op.nn.conv2d_transpose(inputs[0], weight_t, **new_attrs)
        else:
            res = _op.nn.conv2d(inputs[0], weight_t, **new_attrs)
        if use_bias:
            res = _op.nn.bias_add(res, bias_t, axis=1)
        return [res]
    ndim = len(inputs[3])
    assert ndim <= 2
    if ndim == 1:
        if transposed:
            res = F.conv_transpose1d(inp, weight, bias, stride, pad,
                                    output_padding, groups, dilation)
        else:
            res = F.conv1d(inp, weight, bias, stride, pad, dilation, groups)
    else:
        if transposed:
            res = F.conv_transpose2d(inp, weight, bias, stride, pad,
                                    output_padding, groups, dilation)
        else:
            res = F.conv2d(inp, weight, bias, stride, pad, dilation, groups)

    return [res]


@register_node_handler("aten::batch_norm")
def aten_batch_norm(inputs, attributes, scope):
    inp, weight, bias, running_mean, running_var = inputs[:5]
    training, momentum, eps = inputs[5:8]
    # assert training is False
    ctx = current_context()
    net = ctx.network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        running_mean = running_mean.detach().cpu().numpy()
        running_var = running_var.detach().cpu().numpy()
        weight = weight.detach().cpu().numpy()
        bias = bias.detach().cpu().numpy()
        shift = (-running_mean / np.sqrt(running_var + eps)) * weight + bias
        scale = weight / np.sqrt(running_var + eps)
        power = np.ones_like(shift)
        layer = net.add_scale(inp, trt.ScaleMode.CHANNEL, shift, scale, power)
        output = layer.get_output(0)
        output.name = scope
        layer.name = scope
        ctx.refit_weight_dict[layer.name] = {
            "type": "BatchNorm",
            "running_mean": inputs[3].__torch2trt_weight_name,
            "running_var": inputs[4].__torch2trt_weight_name,
            "weight": inputs[1].__torch2trt_weight_name,
            "bias": inputs[2].__torch2trt_weight_name,
            "eps": eps,
        }
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        running_mean = running_mean.detach().cpu().numpy()
        running_var = running_var.detach().cpu().numpy()
        weight = weight.detach().cpu().numpy()
        bias = bias.detach().cpu().numpy()
        running_mean_t = _expr.var(
            scope + "/running_mean", shape=running_mean.shape, dtype="float32")
        running_var_t = _expr.var(
            scope + "/running_var", shape=running_var.shape, dtype="float32")
        weight_t = _expr.var(
            scope + "/weight", shape=weight.shape, dtype="float32")
        bias_t = _expr.var(scope + "/bias", shape=bias.shape, dtype="float32")
        ctx.tvm_weight_dict[running_mean_t] = running_mean
        ctx.tvm_weight_dict[running_var_t] = running_var
        ctx.tvm_weight_dict[weight_t] = weight
        ctx.tvm_weight_dict[bias_t] = bias
        ctx.refit_weight_dict[running_mean_t.name_hint] = inputs[3].__torch2trt_weight_name
        ctx.refit_weight_dict[running_var_t.name_hint] = inputs[4].__torch2trt_weight_name
        ctx.refit_weight_dict[weight_t.name_hint] = inputs[1].__torch2trt_weight_name
        ctx.refit_weight_dict[bias_t.name_hint] = inputs[2].__torch2trt_weight_name
        new_attrs = {}
        new_attrs["axis"] = 1
        new_attrs["epsilon"] = eps
        new_attrs["center"] = True
        new_attrs["scale"] = True
        new_attrs['gamma'] = weight_t
        new_attrs['beta'] = bias_t
        new_attrs['moving_mean'] = running_mean_t
        new_attrs['moving_var'] = running_var_t
        result, moving_mean, moving_var = _op.nn.batch_norm(inp, **new_attrs)
        return [result]
    res = F.batch_norm(inp, running_mean, running_var, weight, bias,
                       bool(training), momentum, eps)
    return [res]


@register_node_handler("aten::addmm")
def aten_addmm(inputs, attributes, scope):
    mat_to_add, mat1, mat2 = inputs[:3]
    beta, alpha = inputs[3:5]
    ctx = current_context()
    net = ctx.network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        assert beta == 1 and alpha == 1
        assert len(mat_to_add.shape) == 1
        inp = mat1
        weight = mat2.t().detach().cpu().numpy()
        bias = mat_to_add.detach().cpu().numpy()
        C = weight.shape[0]
        # use fc to implement this
        if len(inp.shape) < 3:
            inp = _trt_reshape(net, inp, [-1, 1, 1], scope + "/reshape")
        layer = net.add_fully_connected(inp, C, weight, bias)
        output = layer.get_output(0)
        output.name = scope
        layer.name = scope
        ctx.refit_weight_dict[layer.name] = {
            "type": "Linear",
            "weight": inputs[2].__torch2trt_weight_name,
            "bias": inputs[0].__torch2trt_weight_name,
        }
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        inp = mat1
        weight = mat2.t().detach().cpu().numpy()
        bias = mat_to_add.detach().cpu().numpy()
        C = weight.shape[0]
        weight_t = _expr.var(
            scope + "/weight", shape=weight.shape, dtype="float32")
        ctx.tvm_weight_dict[weight_t] = weight
        ctx.refit_weight_dict[weight_t.name_hint] = inputs[2].__torch2trt_weight_name
        bias_t = _expr.var(
            scope + "/bias", shape=bias.shape, dtype="float32")
        ctx.tvm_weight_dict[bias_t] = bias
        ctx.refit_weight_dict[bias_t.name_hint] = inputs[0].__torch2trt_weight_name
        res = _op.nn.dense(inp, weight_t, units=C)
        res = _op.nn.bias_add(res, bias_t, axis=1)
        return [res]

    res = torch.addmm(beta, mat_to_add, alpha, mat1, mat2)
    return [res]


@register_node_handler("aten::matmul")
def aten_matmul(inputs, attributes, scope):
    mat1, mat2 = inputs[:2]
    ctx = current_context()
    net = ctx.network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        assert isinstance(mat2, torch.Tensor)
        inp = mat1
        weight = mat2.t().detach().cpu().numpy()
        C = weight.shape[0]
        # use fc to implement this
        if len(inp.shape) < 3:
            inp = _trt_reshape(net, inp, [-1, 1, 1], scope + "/reshape")
        layer = net.add_fully_connected(inp, C, weight, trt.Weights())
        output = layer.get_output(0)
        output.name = scope
        layer.name = scope
        ctx.refit_weight_dict[layer.name] = {
            "type": "Linear",
            "weight": inputs[1].__torch2trt_weight_name,
        }
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        inp = mat1
        weight = mat2.t().detach().cpu().numpy()
        C = weight.shape[0]
        weight_t = _expr.var(
            scope + "/weight", shape=weight.shape, dtype="float32")
        ctx.tvm_weight_dict[weight_t] = weight
        ctx.refit_weight_dict[weight_t.name_hint] = inputs[1].__torch2trt_weight_name
        res = _op.nn.dense(inputs[0], weight_t, units=C)
        return [res]
    res = torch.matmul(mat1, mat2)
    return [res]


@register_node_handler("aten::max_pool2d")
def aten_max_pool2d(inputs, attributes, scope):
    inp = inputs[0]
    ksize, stride, pad, dilation, ceil_mode = inputs[1:6]
    if len(stride) == 0:
        stride = ksize
    ctx = current_context()
    net = ctx.network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        layer = net.add_pooling(inp, trt.PoolingType.MAX, ksize)
        layer.stride = stride
        layer.padding = pad
        assert all(
            [b == 1 for b in dilation]), "trt pool don't support dilation"
        output = layer.get_output(0)
        output.name = scope
        layer.name = scope
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        assert all(
            [b == 1 for b in dilation]), "tvm maxpool don't support dilation"
        new_attrs = {}
        new_attrs["pool_size"] = ksize
        new_attrs["strides"] = stride
        new_attrs["padding"] = pad
        new_attrs["ceil_mode"] = ceil_mode
        return [_op.nn.max_pool2d(inp, **new_attrs)]

    res = F.max_pool2d(inp, ksize, stride, pad, dilation, bool(ceil_mode))
    return [res]


@register_node_handler("aten::avg_pool2d")
def aten_avg_pool2d(inputs, attributes, scope):
    inp = inputs[0]
    ksize, stride, pad, ceil_mode, count_include_pad = inputs[1:6]
    if len(stride) == 0:
        stride = ksize
    ctx = current_context()
    net = ctx.network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        layer = net.add_pooling(inp, trt.PoolingType.AVERAGE, ksize)
        layer.stride = stride
        layer.padding = pad
        layer.average_count_excludes_padding = not count_include_pad
        output = layer.get_output(0)
        output.name = scope
        layer.name = scope
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        new_attrs = {}
        new_attrs["pool_size"] = ksize
        new_attrs["strides"] = stride
        new_attrs["padding"] = pad
        new_attrs["ceil_mode"] = ceil_mode
        new_attrs["count_include_pad"] = count_include_pad
        return [_op.nn.avg_pool2d(inp, **new_attrs)]

    inp = inputs[0]
    ksize, stride, pad, ceil_mode = inputs[1:5]
    res = F.avg_pool2d(inp, ksize, stride, pad, bool(ceil_mode),
                       bool(count_include_pad))
    return [res]


@register_node_handler("aten::adaptive_avg_pool2d")
def aten_adaptive_avg_pool2d(inputs, attributes, scope):
    inp = inputs[0]
    ksize = inputs[1]
    ctx = current_context()
    net = ctx.network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        inp_shape = inp.shape[1:]
        ksize = [i // k for i, k in zip(inp_shape, ksize)]
        assert all([i % k == 0 for i, k in zip(inp_shape, ksize)])
        layer = net.add_pooling(inp, trt.PoolingType.AVERAGE, ksize)
        # print("WARNING: adaptive_avg_pool2d support is imcomplete")
        output = layer.get_output(0)
        output.name = scope
        layer.name = scope
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        inp_shape = infer_shape(inp)
        inp_shape = inp_shape[2:]
        ksize = [i // k for i, k in zip(inp_shape, ksize)]
        assert all([i % k == 0 for i, k in zip(inp_shape, ksize)])

        return [_op.nn.avg_pool2d(inp, ksize)]

    inp = inputs[0]
    ksize = inputs[1]
    res = F.adaptive_avg_pool2d(inp, ksize)
    return [res]


@register_node_handler("aten::dropout")
def aten_dropout(inputs, attributes, scope):
    inp = inputs[0]
    ctx = current_context()
    net = ctx.network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        return [inputs[0]]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        return [inputs[0]]
    rate, training = inputs[1:3]
    res = F.dropout2d(inp, rate, bool(training))
    return [res]

@register_node_handler("aten::dropout_")
def aten_dropout_(inputs, attributes, scope):
    inp = inputs[0]
    ctx = current_context()
    net = ctx.network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        return [inputs[0]]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        return [inputs[0]]
    rate, training = inputs[1:3]
    res = F.dropout2d(inp, rate, bool(training), inplace=True)
    return [res]


@register_node_handler("aten::cat")
def aten_cat(inputs, attributes, scope):
    tensors, dim = inputs
    ctx = current_context()
    net = ctx.network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        assert dim > 0
        layer = net.add_concatenation(tensors)
        layer.axis = dim - 1  # trt don't support batch axis
        output = layer.get_output(0)
        output.name = scope
        layer.name = scope
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        return [_op.concatenate(tuple(tensors), axis=dim)]

    res = torch.cat(tensors, dim=dim)
    return [res]


def _trt_torch_slice(net, inp, dim, start, end, step, name):
    ndim = len(inp.shape)
    starts = [0] * ndim
    out_shapes = [0] * ndim
    steps = [0] * ndim
    for i in range(ndim):
        starts[i] = 0
        out_shapes[i] = inp.shape[i]
        steps[i] = 1
    starts[dim - 1] = start
    out_shapes[dim - 1] = min(end, inp.shape[dim - 1]) - start
    steps[dim - 1] = step
    layer = net.add_slice(inp, tuple(starts), tuple(out_shapes), tuple(steps))
    output = layer.get_output(0)
    layer.name = name
    return output


def _tvm_torch_slice(inp, dim, start, end, step, name):
    inp_shape = infer_shape(inp)
    ndim = len(inp_shape)
    starts = [0] * ndim
    ends = [0] * ndim
    steps = [0] * ndim
    for i in range(ndim):
        starts[i] = 0
        ends[i] = inp_shape[i]
        steps[i] = 1
    starts[dim] = start
    ends[dim] = min(end, int(inp_shape[dim]))
    steps[dim] = step
    new_attrs = {'begin': starts, 'end': ends, "strides": steps}
    return _op.strided_slice(inp, **new_attrs)


@register_node_handler("aten::slice")
def aten_slice(inputs, attributes, scope):
    inp, dim, start, end, step = inputs
    ctx = current_context()
    net = ctx.network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        if dim == 0:
            if start == 0 and step == 1 and end > 100000000:
                return [inp]
            else:
                assert dim > 0, "tensorrt don't support batch axis operation"
        assert step == 1
        output = _trt_torch_slice(net, inp, dim, start, end, step, scope)
        output.name = scope
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        return [_tvm_torch_slice(inp, dim, start, end, step, scope)]
    slice_ = slice(start, end, step)
    slices = [slice(None, None, None) for _ in range(dim + 1)]
    slices[dim] = slice_
    # res = torch.slice(inp, dim, start, end, step)
    return [inp[slices]]


def _trt_squeeze(net, inp, dim, name):
    assert dim > 0
    assert inp.shape[dim - 1] == 1
    shape = list(inp.shape)
    shape[dim - 1] = None
    shape = list([s for s in shape if s is not None])
    layer = net.add_shuffle(inp)
    layer.reshape_dims = shape
    output = layer.get_output(0)
    layer.name = name
    return output


def _trt_unsqueeze(net, inp, dim, name):
    assert dim > 0
    shape = list(inp.shape)
    shape.insert(dim - 1, 1)
    layer = net.add_shuffle(inp)
    layer.reshape_dims = shape
    output = layer.get_output(0)
    layer.name = name
    return output


def _trt_reshape(net, inp, shape, name):
    layer = net.add_shuffle(inp)
    layer.reshape_dims = shape
    output = layer.get_output(0)
    layer.name = name
    return output


def _tvm_squeeze(inp, dim, name):
    return _op.squeeze(inp, axis=[dim])


def _tvm_unsqueeze(inp, dim, name):
    inp_shape = infer_shape(inp)
    inp_shape = list(inp_shape)
    inp_shape.insert(dim, 1)
    return _op.reshape(inp, newshape=inp_shape)


def _tvm_reshape(inp, shape, name):
    return _op.reshape(inp, newshape=shape)


@register_node_handler("aten::unsqueeze")
def aten_unsqueeze(inputs, attributes, scope):
    inp, dim = inputs
    ctx = current_context()
    net = ctx.network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        return [_trt_unsqueeze(net, inp, dim, scope)]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        return [_tvm_unsqueeze(inp, dim, scope)]
    return [inp.unsqueeze(dim)]


@register_node_handler("aten::select")
def aten_select(inputs, attributes, scope):
    inp, dim, index = inputs
    ctx = current_context()
    net = ctx.network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        assert dim > 0
        output = _trt_torch_slice(net, inp, dim, index, index + 1, 1,
                                  scope + "/slice")
        output.name = scope + "/slice"
        output = _trt_squeeze(net, output, dim, scope + "/squeeze")
        output.name = scope + "/squeeze"
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        output = _tvm_torch_slice(inp, dim, index, index + 1, 1,
                                  scope + "/slice")
        output = _tvm_squeeze(output, dim, scope + "/squeeze")
        return [output]
    slice_ = slice(index, index + 1, 1)
    slices = [slice(None, None, None) for _ in range(dim + 1)]
    slices[dim] = slice_
    return [inp[slices].squeeze(dim)]


def _scale_or_elementwise(net, lfs, rfs, op, name):
    """pytorch elementwise may contains constants.
    if contains constant, use add_scale, otherwise use add_elementwise
    """
    trt_op = {
        "add": trt.ElementWiseOperation.SUM,
        "sub": trt.ElementWiseOperation.SUB,
        "mul": trt.ElementWiseOperation.PROD,
        "div": trt.ElementWiseOperation.DIV,
    }
    assert op in trt_op
    assert not all([isinstance(t, torch.Tensor) for t in [lfs, rfs]])
    if all([isinstance(t, trt.ITensor) for t in [lfs, rfs]]):
        layer = net.add_elementwise(lfs, rfs, trt_op[op])
        layer.name = name
        output = layer.get_output(0)
        return output
    if isinstance(rfs, torch.Tensor):
        val = rfs.detach().cpu().numpy()
        main = lfs
        scale = val
        if val.size == 1:
            # use scale implementation
            if op == "add":
                shift = trt.Weights(np.array(scale, dtype=np.float32))
                scale = trt.Weights(np.array(1, dtype=np.float32))
            elif op == "sub":
                shift = trt.Weights(np.array(-scale, dtype=np.float32))
                scale = trt.Weights(np.array(1, dtype=np.float32))
            elif op == "mul":
                shift = trt.Weights(np.array(0, dtype=np.float32))
                scale = trt.Weights(np.array(scale, dtype=np.float32))
            elif op == "div":
                shift = trt.Weights(np.array(0, dtype=np.float32))
                scale = trt.Weights(np.array(1 / scale, dtype=np.float32))
            else:
                raise NotImplementedError
            power = trt.Weights(np.array(1, dtype=np.float32))
            layer = net.add_scale(main, trt.ScaleMode.UNIFORM, shift, scale,
                                  power)
        else:
            lfs, rfs = try_convert_to_constant(net, [lfs, rfs])
            layer = net.add_elementwise(lfs, rfs, trt_op[op])
    else:
        lfs, rfs = try_convert_to_constant(net, [lfs, rfs])
        layer = net.add_elementwise(lfs, rfs, trt_op[op])
    layer.name = name
    output = layer.get_output(0)
    return output


def try_convert_to_constant(net, inputs):
    res = []
    ref_shape = None
    for inp in inputs:
        if isinstance(inp, trt.ITensor):
            ref_shape = inp.shape
    for inp in inputs:
        if isinstance(inp, torch.Tensor):
            inp = inp.detach().cpu().numpy()
            if inp.dtype == np.float64:
                inp = inp.astype(np.float32)
            if len(inp.shape) == 0:
                inp = inp.reshape(*([1] * len(ref_shape)))
            layer = net.add_constant(inp.shape, trt.Weights(inp))
            inp = layer.get_output(0)
        res.append(inp)
    return res


def torchdtype_to_tvm_map():
    return {
        torch.float32: "float32",
        torch.float64: "float64",
        torch.float16: "float16",
        torch.int32: "int32",
        torch.int64: "int64",
        torch.uint8: "uint8",
    }


def torchdtype_to_tvm_map_inv():
    dict_ = torchdtype_to_tvm_map()
    return {v: k for k, v in dict_.items()}


def torchdtype_to_tvm(ttype):
    return torchdtype_to_tvm_map()[ttype]


def tvm_to_torchdtype(dtype):
    return torchdtype_to_tvm_map_inv()[dtype]


def _tvm_to_const(args):
    ref_type = None
    for arg in args:
        if isinstance(arg, _expr.Expr):
            ref_type = infer_dtype(arg)
    ttype = tvm_to_torchdtype(ref_type)
    res = []
    for arg in args:
        if isinstance(arg, torch.Tensor):
            dtype = torchdtype_to_tvm(arg.dtype)
            arg = _expr.const(
                arg.to(ttype).detach().cpu().float().numpy(), dtype=dtype)
        elif not isinstance(arg, _expr.Expr):
            arg = _expr.const(arg, dtype=dtype)
        res.append(arg)
    return res


@register_node_handler("aten::mul")
def aten_mul(inputs, attributes, scope):
    # print_inputs(inputs)
    lfs, rfs = inputs
    ctx = current_context()
    net = ctx.network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        output = _scale_or_elementwise(net, lfs, rfs, "mul", scope)
        output.name = scope
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        lfs, rfs = _tvm_to_const([lfs, rfs])
        return [_op.multiply(lfs, rfs)]
    return [lfs * rfs]


@register_node_handler("aten::mul_")
def aten_mul_(inputs, attributes, scope):
    # print_inputs(inputs)
    lfs, rfs = inputs
    ctx = current_context()
    net = ctx.network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        output = _scale_or_elementwise(net, lfs, rfs, "mul", scope)
        output.name = scope
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        lfs, rfs = _tvm_to_const([lfs, rfs])
        return [_op.multiply(lfs, rfs)]

    lfs *= rfs
    return [lfs]


@register_node_handler("aten::add_")
def aten_add_(inputs, attributes, scope):
    lfs, rfs, alpha = inputs
    assert alpha == 1
    ctx = current_context()
    net = ctx.network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        output = _scale_or_elementwise(net, lfs, rfs, "add", scope)
        output.name = scope
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        lfs, rfs = _tvm_to_const([lfs, rfs])
        return [_op.add(lfs, rfs)]

    lfs.add_(rfs)
    return [lfs]


@register_node_handler("aten::add")
def aten_add(inputs, attributes, scope):
    lfs, rfs, alpha = inputs
    assert alpha == 1
    ctx = current_context()
    net = ctx.network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        output = _scale_or_elementwise(net, lfs, rfs, "add", scope)
        output.name = scope
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        lfs, rfs = _tvm_to_const([lfs, rfs])
        return [_op.add(lfs, rfs)]
    return [lfs + rfs]


@register_node_handler("aten::div")
def aten_div(inputs, attributes, scope):
    # print_inputs(inputs)
    lfs, rfs = inputs
    ctx = current_context()
    net = ctx.network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        output = _scale_or_elementwise(net, lfs, rfs, "div", scope)
        output.name = scope
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        lfs, rfs = _tvm_to_const([lfs, rfs])
        return [_op.divide(lfs, rfs)]
    return [lfs / rfs]


@register_node_handler("aten::sub")
def aten_sub(inputs, attributes, scope):
    # print_inputs(inputs)
    lfs, rfs, alpha = inputs
    assert alpha == 1
    ctx = current_context()
    net = ctx.network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        output = _scale_or_elementwise(net, lfs, rfs, "sub", scope)
        output.name = scope
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        lfs, rfs = _tvm_to_const([lfs, rfs])
        return [_op.subtract(lfs, rfs)]

    return [lfs - rfs]


def _axes_to_trt_axis(axes, ndim):
    bit = np.array(1, dtype=np.uint32)
    res = np.array(0, dtype=np.uint32)
    for ax in axes:
        if ax == -1:
            ax = ndim
        assert ax > 0
        res = np.bitwise_or(res, bit << (ax - 1))
    return int(res)


@register_node_handler("aten::sum")
def aten_sum(inputs, attributes, scope):
    inp, dim, keepdim = inputs[:3]
    ctx = current_context()
    net = ctx.network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        if not isinstance(dim, list):
            dim = [dim]
        axis_trt = _axes_to_trt_axis(dim, len(inp.shape))
        layer = net.add_reduce(inp, trt.ReduceOperation.SUM, axis_trt,
                               bool(keepdim))
        output = layer.get_output(0)
        output.name = scope
        layer.name = scope
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        return [_op.reduce.sum(inp, dim, keepdims=bool(keepdim))]

    return [inp.sum(dim, keepdim=bool(keepdim))]


@register_node_handler("aten::topk")
def aten_topk(inputs, attributes, scope):
    inp, k, dim = inputs[:3]
    ctx = current_context()
    net = ctx.network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        if not isinstance(dim, list):
            dim = [dim]
        axis_trt = _axes_to_trt_axis(dim, len(inp.shape))
        layer = net.add_topk(inp, trt.TopKOperation.MAX, k, axis_trt)
        output0 = layer.get_output(0)
        output1 = layer.get_output(1)
        output0.name = scope + "_val"
        output1.name = scope + "_inds"
        layer.name = scope
        return [output0, output1]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        raise NotImplementedError

    return inp.topk(k, dim)


@register_node_handler("aten::max")
def aten_max(inputs, attributes, scope):
    inp, dim, keepdim = inputs[:3]
    ctx = current_context()
    net = ctx.network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        if not isinstance(dim, list):
            dim = [dim]
        axis_trt = _axes_to_trt_axis(dim, len(inp.shape))
        layer = net.add_reduce(inp, trt.ReduceOperation.MAX, axis_trt,
                               bool(keepdim))
        output = layer.get_output(0)
        layer.name = scope
        output.name = scope
        return [output, None]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        return [_op.reduce.max(inp, dim, keepdims=bool(keepdim)), None]

    return [*inp.max(dim, keepdim=bool(keepdim))]


@register_node_handler("aten::min")
def aten_min(inputs, attributes, scope):
    inp, dim, keepdim = inputs[:3]
    ctx = current_context()
    net = ctx.network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        if not isinstance(dim, list):
            dim = [dim]
        axis_trt = _axes_to_trt_axis(dim, len(inp.shape))
        layer = net.add_reduce(inp, trt.ReduceOperation.MIN, axis_trt,
                               bool(keepdim))
        output = layer.get_output(0)
        layer.name = scope
        output.name = scope
        return [output, None]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        return [_op.reduce.min(inp, dim, keepdims=bool(keepdim)), None]

    return [*inp.min(dim, keepdim=bool(keepdim))]


@register_node_handler("aten::mean")
def aten_mean(inputs, attributes, scope):
    inp, dim, keepdim = inputs[:3]
    ctx = current_context()
    net = ctx.network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        if not isinstance(dim, list):
            dim = [dim]
        axis_trt = _axes_to_trt_axis(dim, len(inp.shape))
        layer = net.add_reduce(inp, trt.ReduceOperation.AVG, axis_trt,
                               bool(keepdim))
        output = layer.get_output(0)
        layer.name = scope
        output.name = scope
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        return [_op.reduce.mean(inp, dim, keepdims=bool(keepdim))]

    return [inp.mean(dim, keepdim=bool(keepdim))]


@register_node_handler("aten::prod")
def aten_prod(inputs, attributes, scope):
    inp, dim, keepdim = inputs[:3]
    ctx = current_context()
    net = ctx.network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        axis_trt = _axes_to_trt_axis([dim], len(inp.shape))
        layer = net.add_reduce(inp, trt.ReduceOperation.PROD, axis_trt,
                               bool(keepdim))
        output = layer.get_output(0)
        layer.name = scope
        output.name = scope
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        return [_op.reduce.prod(inp, dim, keepdims=bool(keepdim))]

    return [inp.prod(dim, keepdim=bool(keepdim))]


@register_node_handler("aten::permute")
def aten_permute(inputs, attributes, scope):
    inp, params = inputs
    ctx = current_context()
    net = ctx.network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        perm_params = params[1:]
        assert all([p > 0 for p in perm_params])
        layer = net.add_shuffle(inp)
        layer.first_transpose = tuple(p - 1 for p in perm_params)
        output = layer.get_output(0)
        output.name = scope
        layer.name = scope
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        return [_op.transform.transpose(inp, params)]
    return [inputs[0].permute(*params)]


@register_node_handler("aten::transpose")
def aten_transpose(inputs, attributes, scope):
    inp, dim0, dim1 = inputs
    ctx = current_context()
    net = ctx.network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        assert all([p > 0 for p in [dim0, dim1]])
        params = list(range(len(inp.shape)))
        tmp = params[dim1 - 1]
        params[dim1 - 1] = params[dim0 - 1]
        params[dim0 - 1] = tmp
        layer = net.add_shuffle(inp)
        layer.first_transpose = tuple(params)
        output = layer.get_output(0)
        output.name = scope
        layer.name = scope
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        inp_shape = infer_shape(inp)
        params = list(range(len(inp_shape)))
        tmp = params[dim1]
        params[dim1] = params[dim0]
        params[dim0] = tmp
        return [_op.transform.transpose(inp, params)]

    return [torch.transpose(inputs[0], dim0, dim1)]


@register_node_handler("aten::chunk")
def aten_chunk(inputs, attributes, scope):
    inp, chunk, dim = inputs
    ctx = current_context()
    net = ctx.network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        assert dim > 0
        # use slice to implement chunk
        outputs = []
        step = inp.shape[dim - 1] // chunk
        for i in range(chunk):
            out = _trt_torch_slice(net, inp, dim, i * step, (i + 1) * step, 1,
                                   scope + "/slice_{}".format(i))
            outputs.append(out)
        return [outputs]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        outputs = []
        shape = infer_shape(inp)
        step = shape[dim] // chunk
        for i in range(chunk):
            out = _tvm_torch_slice(inp, dim, i * step, (i + 1) * step, 1,
                                   scope + "/slice_{}".format(i))
            outputs.append(out)
        return [outputs]

    return [torch.chunk(inputs[0], chunk, dim)]


@register_node_handler("aten::contiguous")
def aten_contiguous(inputs, attributes, scope):
    ctx = current_context()
    net = ctx.network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        return [inputs[0]]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        return [inputs[0]]
    return [inputs[0].contiguous()]


@register_node_handler("aten::constant_pad_nd")
def aten_constant_pad_nd(inputs, attributes, scope):
    inp, pad_params, val = inputs
    ctx = current_context()
    net = ctx.network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        assert val == 0.0
        w0, h0, w1, h1 = pad_params
        layer = net.add_padding(inp, (w0, h0), (w1, h1))
        output = layer.get_output(0)
        output.name = scope
        layer.name = scope
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        w0, h0, w1, h1 = pad_params
        pad_width = [(0, 0), (0, 0), (w0, h0), (w1, h1)]
        return [_op.nn.pad(inp, pad_width, val)]

    return [F.pad(inp, pad_params, value=val)]


@register_node_handler("aten::softmax")
def aten_softmax(inputs, attributes, scope):
    inp, axis, dtype = inputs[:3]
    ctx = current_context()
    net = ctx.network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        axes_trt = _axes_to_trt_axis([axis], len(inp.shape))
        layer = net.add_softmax(inp)
        layer.axes = axes_trt
        output = layer.get_output(0)
        output.name = scope
        layer.name = scope
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        return [_op.nn.softmax(inputs[0], axis=axis)]
    return [F.softmax(inp, axis, dtype)]


@register_node_handler("aten::index_select")
def aten_index_select(inputs, attributes, scope):
    inp, axis, index = inputs
    ctx = current_context()
    net = ctx.network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        layer = net.add_gather(inp, index, axis - 1)
        output = layer.get_output(0)
        output.name = scope
        layer.name = scope
        return [output]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        raise NotImplementedError
    return [torch.index_select(inp, axis, index)]


@register_node_handler("aten::repeat")
def aten_repeat(inputs, attributes, scope):
    inp, params = inputs
    ctx = current_context()
    net = ctx.network
    if ctx.is_tensorrt and has_trt_tensor(inputs):
        assert params[0] == 1
        assert len(params) > 1
        # assert len(params) == len(inp.shape) + 1
        # implement repeat by several gather operation, slower than native repeat
        i = 0
        for p, s in zip(params[1:], inp.shape):
            if p > 1:
                repeat_weights = np.tile(np.arange(0, s), [p]).astype(np.int32)
                layer = net.add_constant([s * p],
                                         trt.Weights(repeat_weights))
                layer.name = scope + "/" + "constant_{}".format(i)
                gather_inds = layer.get_output(0)
                gather_inds.name = scope + "/" + "constant_{}".format(i)
                layer = net.add_gather(inp, gather_inds, i)
                layer.name = scope + "/" + "gather_{}".format(i)
                out = layer.get_output(0)
                out.name = scope + "/" + "gather_{}".format(i)
            i += 1
        return [out]
    elif ctx.is_tvm and has_tvm_tensor(inputs):
        raise NotImplementedError

    return [inp.repeat(*params)]
