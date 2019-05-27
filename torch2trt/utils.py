import torch 
import tensorrt as trt

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
    else:
        msg += str(obj)
    return msg