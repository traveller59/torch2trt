"""
some functions copy from https://github.com/pytorch/pytorch/blob/master/torch/utils/tensorboard/_pytorch_graph.py
"""

import contextlib
import logging
import re
import time
from collections import OrderedDict

import numpy as np
import tensorrt as trt
import torch
from torch.jit import _unique_state_dict
from torch.onnx.utils import \
    OperatorExportTypes  # must import this, otherwise error

from torch2trt.utils import print_inputs, pretty_str, get_torch_forward_name

TVM_ENABLE = True 
try:
    import tvm 
    from tvm.relay import expr as _expr
    from tvm.relay import op as _op
    from tvm import nd as _nd

except ImportError:
    TVM_ENABLE = False

def tvm_enable():
    return TVM_ENABLE

methods_OP = [
    'attributeNames', 'hasMultipleOutputs', 'hasUses', 'inputs', 'kind',
    'outputs', 'outputsSize', 'scopeName'
]
# Some additional methods to explure for methods_IO are
#
#   'unique' (type int)
#   'type' (type <Tensor<class 'torch._C.Type'>>)
#
# But the below are sufficient for now.
methods_IO = ['node', 'offset', 'debugName']

REGISTERED_NODE_HANDLERS = {}

class GlobalContext:
    def __init__(self, network: trt.INetworkDefinition):
        self.network = network
        self.refit_weight_dict = {} # contains weight name map for trt engine refit
        self.tvm_weight_dict = OrderedDict() # contains tvm var to tvm ndarray
        self.torch_weight_nodes_dict = {}
        self.current_node = None

    @property
    def is_tensorrt(self):
        return isinstance(self.network, trt.INetworkDefinition)

    @property
    def is_torch(self):
        return self.network is None 

    @property
    def is_tvm(self):
        return self.network == "tvm"


CURRENT_CONTEXT = GlobalContext(None)  # when None, will use pytorch debug mode.

@contextlib.contextmanager
def trt_network(net):
    global CURRENT_CONTEXT
    backup = CURRENT_CONTEXT
    CURRENT_CONTEXT = GlobalContext(net)
    yield None
    CURRENT_CONTEXT = backup


@contextlib.contextmanager
def torch_network():
    global CURRENT_CONTEXT
    backup = CURRENT_CONTEXT
    CURRENT_CONTEXT = GlobalContext(None)
    yield None
    CURRENT_CONTEXT = backup

@contextlib.contextmanager
def tvm_network():
    global CURRENT_CONTEXT
    backup = CURRENT_CONTEXT
    CURRENT_CONTEXT = GlobalContext("tvm")
    yield None
    CURRENT_CONTEXT = backup

def current_context() -> GlobalContext:
    global CURRENT_CONTEXT
    return CURRENT_CONTEXT


def register_node_handler(name):
    def wrap_func(handler):
        global REGISTERED_NODE_HANDLERS
        # assert name not in REGISTERED_NODE_HANDLERS, f"exist handlers: {REGISTERED_NODE_HANDLERS.keys()}"
        REGISTERED_NODE_HANDLERS[name] = handler
        def new_handler(inputs, attributes, scope):
            return handler(inputs, attributes, scope)
        return new_handler

    return wrap_func


def get_node_handler(name):
    global REGISTERED_NODE_HANDLERS
    msg = "missing handler " + name
    msg += ", available handlers: {}".format(list(REGISTERED_NODE_HANDLERS.keys()))
    assert name in REGISTERED_NODE_HANDLERS, msg
    return REGISTERED_NODE_HANDLERS[name]


def has_trt_tensor(inputs):
    res = False
    for inp in inputs:
        if isinstance(inp, (list, tuple)):
            for elem in inp:
                res |= has_trt_tensor([elem])
        elif isinstance(inp, trt.ITensor):
            return True
    return res


def has_torch_tensor(inputs):
    res = False
    for inp in inputs:
        if isinstance(inp, (list, tuple)):
            for elem in inp:
                res |= has_torch_tensor([elem])
        elif isinstance(inp, torch.Tensor):
            return True
    return res

def has_tvm_tensor(inputs):
    if not tvm_enable():
        return False
    res = False
    for inp in inputs:
        if isinstance(inp, (list, tuple)):
            for elem in inp:
                res |= has_tvm_tensor([elem])
        elif isinstance(inp, _expr.Expr):
            return True
    return res

def have_tensor(inputs):
    return has_tvm_tensor(inputs) or has_trt_tensor(inputs) or has_torch_tensor(inputs)

class NodeBase(object):
    def __init__(self,
                 debugName=None,
                 inputs=None,
                 scope=None,
                 tensor_size=None,
                 op_type='UnSpecified',
                 attributes=''):
        # TODO; Specify a __slots__ for this class or potentially
        # used namedtuple instead
        self.debugName = debugName
        self.inputs = inputs
        self.tensor_size = tensor_size
        self.kind = op_type
        self.attributes = attributes
        self.scope = scope

    def __repr__(self):
        repr = []
        repr.append(str(type(self)))
        for m in dir(self):
            if m == "resolved_outputs":
                resolved_outputs = getattr(self, m)
                reprs = []
                for v in resolved_outputs:
                    if isinstance(v, torch.Tensor):
                        reprs.append("Tensor|dtype={}|shape={}".format(
                            v.dtype, list(v.shape)))
                    else:
                        reprs.append(str(v))
                repr.append(m + ': [' + ','.join(reprs) + ']')
            else:
                if '__' not in m:
                    repr.append(m + ': ' + str(getattr(self, m)) +
                                str(type(getattr(self, m))))
        return '\n'.join(repr) + '\n\n'


class NodePy(NodeBase):
    def __init__(self, node_cpp, valid_methods):
        super(NodePy, self).__init__(node_cpp)
        valid_methods = valid_methods[:]
        self.inputs = []
        self.resolved_outputs = []
        self.resolved = False
        self.readable_unique_name = None

        for m in valid_methods:
            if m == 'inputs' or m == 'outputs':
                list_of_node = list(getattr(node_cpp, m)())
                io_unique_names = []
                io_tensor_sizes = []
                for n in list_of_node:
                    io_unique_names.append(n.debugName())
                    if n.type().kind() == 'CompleteTensorType':
                        io_tensor_sizes.append(n.type().sizes())
                    else:
                        io_tensor_sizes.append(None)

                setattr(self, m, io_unique_names)
                setattr(self, m + 'tensor_size', io_tensor_sizes)

            else:
                setattr(self, m, getattr(node_cpp, m)())
        self.resolved_inputs = [None] * len(self.inputs)


class NodePyIO(NodePy):
    def __init__(self, node_cpp, input_or_output=None):
        super(NodePyIO, self).__init__(node_cpp, methods_IO)
        try:
            tensor_size = node_cpp.type().sizes()
        except RuntimeError:
            tensor_size = [
                1,
            ]  # fail when constant model is used.
        self.tensor_size = tensor_size
        # Kind attribute string is purely descriptive and will be shown
        # in detailed information for the node in TensorBoard's graph plugin.
        #
        # NodePyOP nodes get this from their kind() method.
        self.kind = 'Parameter'
        if input_or_output:
            self.input_or_output = input_or_output
            self.kind = 'IO Node'
        self.resolved_outputs = [None]
        self.resolved = True
        self._weight_name = None


class NodePyOP(NodePy):
    def __init__(self, node_cpp):
        super(NodePyOP, self).__init__(node_cpp, methods_OP)
        # Replace single quote which causes strange behavior in TensorBoard
        # TODO: See if we can remove this in the future
        # self.attributes = str({k: node_cpp[k] for k in node_cpp.attributeNames()}).replace("'", ' ')
        self.attributes = {k: node_cpp[k] for k in node_cpp.attributeNames()}
        self.kind = node_cpp.kind()
        self.resolved_outputs = [None] * len(self.outputs)


class GraphPy(object):
    """Helper class to convert torch.nn.Module to GraphDef proto and visualization
    with TensorBoard.
    GraphDef generation operates in two passes:
    In the first pass, all nodes are read and saved to two lists.
    One list is for input/output nodes (nodes_io), which only have inbound
    or outbound connections, but not both. Another list is for internal
    operator nodes (nodes_op). The first pass also saves all scope name
    appeared in the nodes in scope_name_appeared list for later processing.
    In the second pass, scope names are fully applied to all nodes.
    uniqueNameToScopedName is a mapping from a node's ID to its fully qualified
    scope name. e.g. Net1/Linear[0]/1. Unfortunately torch.jit doesn't have
    totally correct scope output, so this is nontrivial. The function
    populate_namespace_from_OP_to_IO and find_common_root are used to
    assign scope name to a node based on the connection between nodes
    in a heuristic kind of way. Bookkeeping is done with shallowest_scope_name
    and scope_name_appeared.
    """

    def __init__(self):
        self.nodes_op = []
        self.nodes_io = OrderedDict()
        self.unique_name_to_scoped_name = {}
        self.shallowest_scope_name = 'default'
        self.scope_name_appeared = []
        self.output_nodes = []
        self.readable_name_to_node = {}
        self.refit_weight_dict = {}
        self.context = None
        self.is_class = False
        self.torch_weight_nodes_dict = set()
        self.unique_name_to_name = {}

    def get_output_nodes_dict(self):
        nodes_dict = OrderedDict()
        for k, node in self.nodes_io.items():
            if hasattr(node, 'input_or_output'):
                if node.input_or_output == "output":
                    nodes_dict[k] = node
        return nodes_dict

    def get_input_nodes_dict(self):
        nodes_dict = OrderedDict()
        for k, node in self.nodes_io.items():
            if hasattr(node, 'input_or_output'):
                if node.input_or_output == "input":
                    nodes_dict[k] = node
        return nodes_dict

    def get_param_nodes_dict(self):
        nodes_dict = OrderedDict()
        for k, node in self.nodes_io.items():
            if isinstance(node, NodePyIO):
                if node.kind == 'Parameter':
                    nodes_dict[k] = node
        return nodes_dict

    def get_param_nodes(self):
        nodes = []
        for k, node in self.nodes_io.items():
            if isinstance(node, NodePyIO):
                if node.kind == 'Parameter':
                    nodes.append(node)
        return nodes

    def get_out_to_node(self):
        out_to_node = {}
        for node in self.nodes_op:
            for out in node.outputs:
                out_to_node[out] = node

        return {
            **out_to_node,
            **self.get_input_nodes_dict(),
            **self.get_param_nodes_dict()
        }

    def get_unique_name_to_node(self):
        unique_name_to_node = {}
        for node in self.nodes_op:
            unique_name_to_node[node.readable_unique_name] = node
        for k, node in self.nodes_io.items():
            unique_name_to_node[node.readable_unique_name] = node
        return unique_name_to_node

    def get_out_to_idx(self):
        out_to_idx = {}
        for node in self.nodes_op:
            for i, out in enumerate(node.outputs):
                out_to_idx[out] = i
        for k, v in self.get_input_nodes_dict().items():
            out_to_idx[k] = 0
        for k, v in self.get_param_nodes_dict().items():
            out_to_idx[k] = 0

        return out_to_idx

    def get_output_names(self):
        return [n.debugName for n in self.output_nodes]

    def get_resolved_outputs(self):
        """return list of list: first list is list of outputs in net.forward,
        second list is list of output in output node because some mode return multiple outputs.
        """
        res = []
        out_to_node = self.get_out_to_node()
        out_to_idx = self.get_out_to_idx()
        for name in self.get_output_names():
            res.append(out_to_node[name].resolved_outputs[out_to_idx[name]])
        if len(res) == 1:
            return res[0]
        return tuple(res)

    def append(self, x):
        if isinstance(x, NodePyIO):
            self.nodes_io[x.debugName] = x
        if isinstance(x, NodePyOP):
            self.nodes_op.append(x)
            for node_output, outputSize in zip(x.outputs,
                                               x.outputstensor_size):
                self.scope_name_appeared.append(x.scopeName)
                self.nodes_io[node_output] = NodeBase(
                    node_output,
                    x.inputs,
                    x.scopeName,
                    outputSize,
                    op_type=x.kind,
                    attributes=x.attributes)

    def printall(self):
        print('all nodes')
        for node in self.nodes_op:
            print(node)
        for key in self.nodes_io:
            print(self.nodes_io[key])

    def get_all_kind(self):
        kinds = set()
        for node in self.nodes_op:
            kinds.add(node.kind)
        for key in self.nodes_io:
            kinds.add(self.nodes_io[key].kind)
        return kinds

    def find_common_root(self):
        for fullscope in self.scope_name_appeared:
            if fullscope:
                self.shallowest_scope_name = fullscope.split('/')[0]

    def populate_namespace_from_OP_to_IO(self):
        for node in self.nodes_op:
            for input_node_id in node.inputs:
                self.unique_name_to_scoped_name[
                    input_node_id] = node.scopeName + '/' + input_node_id

        for key, node in self.nodes_io.items():
            if hasattr(node, 'input_or_output'):
                self.unique_name_to_scoped_name[
                    key] = node.input_or_output + '/' + node.debugName
            if hasattr(node, 'scope') and node.scope is not None:
                self.unique_name_to_scoped_name[
                    key] = node.scope + '/' + node.debugName
                if node.scope == '' and self.shallowest_scope_name:
                    self.unique_name_to_scoped_name[
                        node.
                        debugName] = self.shallowest_scope_name + '/' + node.debugName

        # replace name
        for key, node in self.nodes_io.items():
            self.nodes_io[key].inputs = [
                self.unique_name_to_scoped_name[node_input_id]
                for node_input_id in node.inputs
            ]
            if node.debugName in self.unique_name_to_scoped_name:
                self.nodes_io[
                    key].debugName = self.unique_name_to_scoped_name[
                        node.debugName]


def _make_unique_name(unique_set, name, max_count=10000):
    if name not in unique_set:
        unique_set.add(name)
        return name
    for i in range(max_count):
        new_name = name + "_{}".format(i)
        if new_name not in unique_set:
            unique_set.add(new_name)
            return new_name
    raise ValueError("max count reached")

class UniqueNamePool:
    def __init__(self, max_count=10000):
        self.max_count = max_count
        self.unique_set = set()

    def __call__(self, name):
        return _make_unique_name(self.unique_set, name)


def parse(graph, num_inputs, omit_useless_nodes=False, is_class=False):
    """This method parses an optimized PyTorch model graph and produces
    a list of nodes and node stats for eventual conversion to TensorBoard
    protobuf format.
    Args:
      graph (PyTorch module): The model to be parsed.
      args (tuple): input tensor[s] for the model.
      omit_useless_nodes (boolean): Whether to remove nodes from the graph.
    """
    n_inputs = num_inputs
    if is_class:
        num_inputs += 1
    scope = {}
    graph_py = GraphPy()
    graph_py.is_class = is_class
    for i, node in enumerate(graph.inputs()):
        if omit_useless_nodes:
            if len(
                    node.uses()
            ) == 0:  # number of user of the node (= number of outputs/ fanout)
                continue
        if i < num_inputs:
            graph_py.append(NodePyIO(node, 'input'))
        # else:
        #     graph_py.append(NodePyIO(node))  # parameter
        # print(node)

    for node in graph.nodes():
        graph_py.append(NodePyOP(node))

    for node in graph.outputs():  # must place last.
        graph_py.output_nodes.append(NodePyIO(node, 'output'))
    graph_py.find_common_root()
    graph_py.populate_namespace_from_OP_to_IO()

    # assign a readable name to each node
    output_names = graph_py.get_output_names()
    name_pool = UniqueNamePool()
    for output_name in output_names:
        out_to_node = graph_py.get_out_to_node()
        out_node = out_to_node[output_name]
        def recursive_assign_name(node: NodePy):
            if isinstance(node, NodePyOP):
                if node.readable_unique_name is None:
                    if node.scopeName != "":
                        name = node.scopeName + "/" + node.kind
                    else:
                        name = node.kind
                    node.readable_unique_name = name_pool(name)
                    node.readable_unique_name = node.readable_unique_name.replace(":", "_")
                    node.readable_unique_name = node.readable_unique_name.replace("/", ".")
                    # node.readable_unique_name = name_pool("layer")
                    graph_py.readable_name_to_node[name] = node
                else:
                    return
            else:
                return
            for inp_name in node.inputs:
                recursive_assign_name(out_to_node[inp_name])
        recursive_assign_name(out_node)
    return graph_py


def resolve_graph(graph_py: GraphPy, output_names, verbose=False):
    ctx = current_context()
    out_to_node = graph_py.get_out_to_node()
    out_to_idx = graph_py.get_out_to_idx()
    if not isinstance(output_names, (list, tuple)):
        output_names = [output_names]
    node_results = []
    for output_name in output_names:
        if not isinstance(output_name, str):
            out_node = output_name
        else:
            out_node = out_to_node[output_name]
        stack = [out_node]
        while len(stack) > 0:
            node = stack[-1]
            if node.resolved:
                stack.pop()
                continue
            
            inputs = []
            input_nodes = []
            prepared = True
            for inp_name in node.inputs:
                inp_node = out_to_node[inp_name]
                if not inp_node.resolved:  # node isn't evaluated
                    stack.append(inp_node)
                    prepared = False
                input_nodes.append(inp_node)
                inputs.append(inp_node.resolved_outputs[out_to_idx[inp_name]])
            if not prepared:
                continue
            assert node.readable_unique_name is not None
            if verbose:
                msg = ""
                if have_tensor(inputs):
                    msg += "{}==>>".format(pretty_str(inputs))
            try:
                handler = get_node_handler(node.kind)
                ctx.current_node = node
                results = handler(inputs, node.attributes,
                                  node.readable_unique_name)
                ctx.current_node = None
            except Exception as e:
                print(node.readable_unique_name)
                if verbose:
                    print(msg)
                raise e
            assert isinstance(results, (list, tuple)), node.kind
            assert len(results) == len(node.resolved_outputs), node.kind
            if verbose:
                if have_tensor(inputs) and have_tensor(results):
                    msg += pretty_str(results)
                    print(node.readable_unique_name)
                    print(msg)
            node.resolved_outputs = list(results)
            node.resolved = True
            stack.pop()
        
        node_results.append(out_node.resolved_outputs)
    graph_py.refit_weight_dict = ctx.refit_weight_dict
    graph_py.torch_weight_nodes_dict = ctx.torch_weight_nodes_dict
    return node_results

def _torch_depoly(module,
              example_inputs,
              output_names=None,
              input_tensors=None,
              input_names=None,
              verbose=False):
    """main entry point of torch2tvm.

    Args:
        module: pytorch nn.Module or function.
        example_inputs: list or tuple of example tensors. MUST match arguments of 
            forward function of module.
        output_names: list of string. indicate output node name you want to use.
            note that pytorch jit node name don't contains any readable information.
            so I recommend not use this.
        input_tensors: list of trt.ITensor. if provided, will use this tensors to evaluate
            graph. otherwise will create new input tensors based on example_inputs
        input_names: list of string. MUST provided when run in trt mode. not required
            in pytorch debug mode.
        verbose: bool. 
    Returns:
        trace: traced jit module or function. MUST returned to avoid some C++ error.
        graph_pth: GraphPy object. use this to access pytorch graph and get
            resolved output tensors.
        tvm_module: 
    """
    trace = torch.jit.trace(module, example_inputs, True)
    if input_names is None:
        if not isinstance(module, torch.nn.Module):
            input_names = get_torch_forward_name(module)
        else:
            input_names = get_torch_forward_name(module.forward)

    if not isinstance(example_inputs, (list, tuple)):
        example_inputs = [example_inputs]
    is_class = isinstance(module, torch.nn.Module)
    graph_py = parse(
        trace.graph, len(example_inputs), omit_useless_nodes=False, is_class=is_class)
    msg = "input mismatch. this may due to some input isn't used in graph"
    assert len(example_inputs) + int(is_class) == len(graph_py.get_input_nodes_dict()), msg
    if output_names is None:
        output_names = graph_py.get_output_names()
    ctx = current_context()
    net = ctx.network
    if ctx.is_tensorrt:
        assert input_names is not None, "trt mode must provide input name"
        if not isinstance(input_names, (list, tuple)):
            input_names = [input_names]
        assert len(input_names) == len(example_inputs)
        inputs = []
        if is_class:
            inputs = [module]
        if input_tensors is not None:
            if not isinstance(input_tensors, (list, tuple)):
                input_tensors = [input_tensors]
            assert len(input_tensors) == len(example_inputs)
            inputs += input_tensors
        else:
            for torch_inp, name in zip(example_inputs, input_names):
                tensor = net.add_input(
                    name=name,
                    dtype=trt.float32,
                    shape=tuple(torch_inp.shape[1:]))
                inputs.append(tensor)
        for i, inode in enumerate(graph_py.get_input_nodes_dict().values()):
            inode.resolved_outputs[0] = inputs[i]
    elif ctx.is_tvm:
        assert input_names is not None, "tvm mode must provide input name"
        if not isinstance(input_names, (list, tuple)):
            input_names = [input_names]
        assert len(input_names) == len(example_inputs)
        inputs = []
        if is_class:
            inputs = [module]
        if input_tensors is not None:
            if not isinstance(input_tensors, (list, tuple)):
                input_tensors = [input_tensors]
            assert len(input_tensors) == len(example_inputs)
            inputs += input_tensors
        else:
            for torch_inp, name in zip(example_inputs, input_names):
                tensor = _expr.var(name, shape=torch_inp.shape, dtype="float32")
                inputs.append(tensor)
        for i, inode in enumerate(graph_py.get_input_nodes_dict().values()):
            inode.resolved_outputs[0] = inputs[i]
    else:
        # use torch inputs, debug mode
        for i, inode in enumerate(graph_py.get_input_nodes_dict().values()):
            inode.resolved_outputs[0] = example_inputs[i]
    resolve_graph(graph_py, output_names, verbose=verbose)
    graph_py.context = ctx
    # trace must be returned to avoid std::bad_alloc
    return trace, graph_py

torch2tvm = _torch_depoly
torch2trt = _torch_depoly

def clean_resolved_outputs(graph_py, output_name):
    if not isinstance(output_name, str):
        out_node = output_name
    else:
        out_to_node = graph_py.get_out_to_node()
        out_node = out_to_node[output_name]

    def recursive_resolve(node: NodePy):
        if isinstance(node, NodePyOP):
            if node.resolved is True:
                node.resolved_outputs = [None] * len(node.resolved_outputs)
                node.resolved = False
            else:
                return
        else:
            return
        for inp_name in node.inputs:
            recursive_resolve(out_to_node[inp_name])

    recursive_resolve(out_node)


class GraphModule:
    """main entry class of torch2trt/torch2tvm.
    Args:
        module: pytorch nn.Module or function.
        example_inputs: list or tuple of example tensors. MUST match arguments of 
            forward function of module.
    """
    def __init__(self,
                 module,
                 example_inputs):

        super().__init__()
        self.module = module
        is_class = isinstance(module, torch.nn.Module)

        trace = torch.jit.trace(module, example_inputs, True)
        if not isinstance(example_inputs, (list, tuple)):
            example_inputs = [example_inputs]
        graph_py = parse(
            trace.graph, len(example_inputs), omit_useless_nodes=False, is_class=is_class)
        self.graph = graph_py
        self.trace = trace
        self.example_inputs = example_inputs

        msg = "input mismatch. this may due to some input isn't used in graph"
        assert len(example_inputs) + int(is_class) == len(graph_py.get_input_nodes_dict()), msg

    def __call__(self, *args, verbose=False, **kw):
        assert len(kw) == 0, "don't support kw arg"
        assert all([isinstance(e, (torch.Tensor, trt.ITensor)) for e in args])
        assert len(args) + int(self.graph.is_class) == len(self.graph.get_input_nodes_dict())
        output_names = self.graph.get_output_names()
        for output_name in output_names:
            clean_resolved_outputs(self.graph, output_name)
        args = list(args)
        arg_for_check = args
        if self.graph.is_class:
            args.insert(0, self.module)
            arg_for_check = args[1:]
        for i, inode in enumerate(self.graph.get_input_nodes_dict().values()):
            inode.resolved_outputs[0] = args[i]
        if has_trt_tensor(arg_for_check):
            # trt mode
            assert all([isinstance(e, trt.ITensor) for e in arg_for_check])
            assert current_context().is_tensorrt
        elif has_tvm_tensor(arg_for_check):
            assert all([isinstance(e, _expr.Expr) for e in arg_for_check])
            assert current_context().is_tvm
        else:
            assert all([isinstance(e, torch.Tensor) for e in arg_for_check])
            assert current_context().is_torch, "you should run pytorch mode outside trt_network block"
        resolve_graph(self.graph, output_names, verbose)
        return self.graph.get_resolved_outputs()

    def collect_params(self, net):
        """rerun graph in pytorch mode and collect new params
        """
        self.graph.torch_weight_nodes_dict = {}
        output_names = self.graph.get_output_names()
        for name in output_names:
            clean_resolved_outputs(self.graph, name)
        with torch.no_grad():
            self(net, *self.example_inputs)
        return self.graph.torch_weight_nodes_dict

    def collect_params_v1(self):
        return self.graph.torch_weight_nodes_dict
