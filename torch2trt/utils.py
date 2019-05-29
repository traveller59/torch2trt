import torch 
import tensorrt as trt
from tvm.relay import ir_pass

TVM_ENABLE = False

try:
    import tvm 
    from tvm.relay import expr as _expr
    TVM_ENABLE = True
except ImportError:
    pass

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
        obj_infer = ir_pass.infer_type(obj)
        dtype = obj_infer.checked_type.dtype
        shape = obj_infer.checked_type.shape
        msg += "TVM|{}|{}".format(dtype, shape)
    else:
        msg += str(obj)
    return msg