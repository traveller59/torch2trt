import torch 

def print_inputs(inputs):
    reprs = []
    for inp in inputs:
        if isinstance(inp, torch.Tensor):
            reprs.append("Tensor|dtype={}|shape={}".format(inp.dtype, list(inp.shape)))
        else:
            reprs.append(str(inp))
    print("[{}]".format(",".join(reprs)))