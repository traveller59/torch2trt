import torch 
import tensorrt as trt
import inspect

TVM_ENABLE = False

try:
    import tvm 
    from tvm.relay import expr as _expr
    from tvm.relay import transform as _transform 
    from tvm.relay import module as _module 
    from topi.util import get_const_tuple
    TVM_ENABLE = True
except ImportError:
    pass

if TVM_ENABLE:
    def _infer_type(node):
        """A method to infer the type of an intermediate node in the relay graph."""
        mod = _module.Module.from_expr(node)
        mod = _transform.InferType()(mod)
        entry = mod["main"]
        return entry if isinstance(node, _expr.Function) else entry.body

    def _infer_shape(node, params=None):
        """A method to get the output shape of an intermediate node in the relay graph."""
        out_type = _infer_type(node)
        return get_const_tuple(out_type.checked_type.shape)

    def _infer_dtype(node, params=None):
        """A method to get the output shape of an intermediate node in the relay graph."""
        out_type = _infer_type(node)
        return out_type.checked_type.dtype

    def infer_shape(tensor):
        return _infer_shape(tensor)

    def infer_dtype(tensor):
        return _infer_dtype(tensor)

def print_inputs(inputs):
    print(pretty_str(inputs))

def pretty_str(obj):
    msg = ""
    if isinstance(obj, list):
        msg += "[" + ",".join([pretty_str(e) for e in obj]) + "]"
    elif isinstance(obj, tuple):
        msg += "(" + ",".join([pretty_str(e) for e in obj]) + ")"
    elif isinstance(obj, dict):
        msg += "{" + ",".join(["{}:{}".format(k, pretty_str(v)) for k, v in obj.items()]) + "}"
    elif isinstance(obj, torch.Tensor):
        msg += "T|{}|{}".format(obj.dtype, tuple(obj.shape))
    elif isinstance(obj, trt.ITensor):
        msg += "TRT|{}|{}".format(obj.dtype, tuple(obj.shape))
    elif TVM_ENABLE and isinstance(obj, _expr.Expr):
        msg += "TVM|{}|{}".format(infer_dtype(obj), infer_shape(obj))
    else:
        msg += str(obj)
    return msg

def get_torch_forward_name(func):
    sig = inspect.signature(func)
    params = sig.parameters # skip self
    res = []
    for k, p in params.items():
        assert p.kind is p.POSITIONAL_OR_KEYWORD
        res.append(k)
    return res
