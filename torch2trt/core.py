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
from torch.utils.tensorboard._proto_graph import node_proto

from torch2trt.utils import print_inputs, pretty_str

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
methods_IO = ['node', 'offset', 'uniqueName']

REGISTERED_NODE_HANDLERS = {}

CURRENT_NETWORK = None  # when None, will use pytorch debug mode.


@contextlib.contextmanager
def trt_network(net):
    global CURRENT_NETWORK
    backup = CURRENT_NETWORK
    CURRENT_NETWORK = net
    yield net
    CURRENT_NETWORK = backup


@contextlib.contextmanager
def torch_network():
    global CURRENT_NETWORK
    backup = CURRENT_NETWORK
    CURRENT_NETWORK = None
    yield None
    CURRENT_NETWORK = backup


def current_network() -> trt.INetworkDefinition:
    global CURRENT_NETWORK
    return CURRENT_NETWORK


def register_node_handler(name):
    def wrap_func(handler):
        global REGISTERED_NODE_HANDLERS
        # assert name not in REGISTERED_NODE_HANDLERS, f"exist handlers: {REGISTERED_NODE_HANDLERS.keys()}"
        REGISTERED_NODE_HANDLERS[name] = handler
        return handler

    return wrap_func


def get_node_handler(name):
    global REGISTERED_NODE_HANDLERS
    msg = "missing handler " + name
    msg += f", available handlers: {REGISTERED_NODE_HANDLERS.keys()}"
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


class NodeBase(object):
    def __init__(self,
                 uniqueName=None,
                 inputs=None,
                 scope=None,
                 tensor_size=None,
                 op_type='UnSpecified',
                 attributes=''):
        # TODO; Specify a __slots__ for this class or potentially
        # used namedtuple instead
        self.uniqueName = uniqueName
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
                    io_unique_names.append(n.uniqueName())
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
        return [n.uniqueName for n in self.output_nodes]

    def get_resolved_outputs(self):
        """return list of list: first list is list of outputs in net.forward,
        second list is list of output in output node because some mode return multiple outputs.
        """
        res = []
        out_to_node = self.get_out_to_node()
        for name in self.get_output_names():
            res.append(out_to_node[name].resolved_outputs)
        return res

    def append(self, x):
        if isinstance(x, NodePyIO):
            self.nodes_io[x.uniqueName] = x
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
                    key] = node.input_or_output + '/' + node.uniqueName
            if hasattr(node, 'scope') and node.scope is not None:
                self.unique_name_to_scoped_name[
                    key] = node.scope + '/' + node.uniqueName
                if node.scope == '' and self.shallowest_scope_name:
                    self.unique_name_to_scoped_name[
                        node.
                        uniqueName] = self.shallowest_scope_name + '/' + node.uniqueName

        # replace name
        for key, node in self.nodes_io.items():
            self.nodes_io[key].inputs = [
                self.unique_name_to_scoped_name[node_input_id]
                for node_input_id in node.inputs
            ]
            if node.uniqueName in self.unique_name_to_scoped_name:
                self.nodes_io[
                    key].uniqueName = self.unique_name_to_scoped_name[
                        node.uniqueName]


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


def parse(graph, num_inputs, omit_useless_nodes=False):
    """This method parses an optimized PyTorch model graph and produces
    a list of nodes and node stats for eventual conversion to TensorBoard
    protobuf format.
    Args:
      graph (PyTorch module): The model to be parsed.
      args (tuple): input tensor[s] for the model.
      omit_useless_nodes (boolean): Whether to remove nodes from the graph.
    """
    n_inputs = num_inputs

    scope = {}
    graph_py = GraphPy()
    for i, node in enumerate(graph.inputs()):
        if omit_useless_nodes:
            if len(
                    node.uses()
            ) == 0:  # number of user of the node (= number of outputs/ fanout)
                continue

        if i < n_inputs:
            graph_py.append(NodePyIO(node, 'input'))
        else:
            graph_py.append(NodePyIO(node))  # parameter

    for node in graph.nodes():
        graph_py.append(NodePyOP(node))

    for node in graph.outputs():  # must place last.
        graph_py.output_nodes.append(NodePyIO(node, 'output'))
    graph_py.find_common_root()
    graph_py.populate_namespace_from_OP_to_IO()

    # assign a readable name to each node
    output_names = graph_py.get_output_names()
    unique_name_set = set()
    for output_name in output_names:
        out_to_node = graph_py.get_out_to_node()
        out_node = out_to_node[output_name]

        def recursive_assign_name(node: NodePy):
            if isinstance(node, NodePyOP):
                if node.readable_unique_name is None:
                    node.readable_unique_name = _make_unique_name(
                        unique_name_set, node.scopeName + "/" + node.kind)
                    graph_py.readable_name_to_node[node.
                                                   readable_unique_name] = node
                else:
                    return
            else:
                return
            for inp_name in node.inputs:
                recursive_assign_name(out_to_node[inp_name])

        recursive_assign_name(out_node)

    return graph_py


def _get_jit_params(module, param_exclude, param_include):
    state_dict = _unique_state_dict(module)
    if param_exclude is not None:
        param_exclude = re.compile(param_exclude)
    if param_include is not None:
        param_include = re.compile(param_include)

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if param_exclude is not None and param_exclude.match(k) is not None:
            continue
        if param_include is not None and param_include.match(k) is None:
            continue
        if "num_batches_tracked" not in k:
            if "weight" in k or "bias" in k or "running_mean" in k or "running_var" in k:
                new_state_dict[k] = v
    params = list(new_state_dict.values())[::-1]
    return params


def resolve_graph(graph_py: GraphPy, output_names, verbose=False):
    out_to_node = graph_py.get_out_to_node()
    out_to_idx = graph_py.get_out_to_idx()
    if not isinstance(output_names, (list, tuple)):
        output_names = [output_names]
    results = []
    for output_name in output_names:
        out_node = out_to_node[output_name]
        stack = [out_node]
        while len(stack) > 0:
            node = stack[-1]
            if node.resolved:
                stack.pop()
                continue
            inputs = []
            prepared = True
            for inp_name in node.inputs:
                inp_node = out_to_node[inp_name]
                if not inp_node.resolved:  # node isn't evaluated
                    stack.append(inp_node)
                    prepared = False
                inputs.append(inp_node.resolved_outputs[out_to_idx[inp_name]])
            if not prepared:
                continue
            assert node.readable_unique_name is not None
            if verbose:
                msg = ""
                if (has_trt_tensor(inputs) or has_torch_tensor(inputs)):
                    msg += "{}==>>".format(pretty_str(inputs))
            try:
                handler = get_node_handler(node.kind)
                results = handler(inputs, node.attributes,
                                  node.readable_unique_name)
            except Exception as e:
                print(node.readable_unique_name)
                print(msg)
                raise e
            assert isinstance(results, (list, tuple)), f"{node.kind}"
            assert len(results) == len(node.resolved_outputs), f"{node.kind}"
            if verbose:
                if ((has_trt_tensor(inputs) or has_torch_tensor(inputs)) and
                    (has_trt_tensor(results) or has_torch_tensor(results))):
                    msg += pretty_str(results)
                    print(node.readable_unique_name)
                    print(msg)
            node.resolved_outputs = list(results)
            node.resolved = True
            stack.pop()
        results.append(out_node.resolved_outputs)
    return results


def torch2trt(module,
              example_inputs,
              param_exclude=None,
              param_include=None,
              output_names=None,
              input_tensors=None,
              input_names=None,
              verbose=False):
    """main entry point of torch2trt.

    Args:
        module: pytorch nn.Module or function.
        example_inputs: list or tuple of example tensors. MUST match arguments of 
            forward function of module.
        param_exclude: regex string. filter unused weights and buffers if match.
        param_include: regex string. filter unused weights and buffers if not match.
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
    """
    trace = torch.jit.trace(module, example_inputs, True)
    if not isinstance(example_inputs, (list, tuple)):
        example_inputs = [example_inputs]
    graph_py = parse(
        trace.graph, len(example_inputs), omit_useless_nodes=False)
    msg = "input mismatch. this may due to some input isn't used in graph"
    assert len(example_inputs) == len(graph_py.get_input_nodes_dict()), msg
    if output_names is None:
        output_names = graph_py.get_output_names()
    if isinstance(module, torch.nn.Module):
        params = _get_jit_params(module, param_exclude, param_include)
        num_param_inputs = len(graph_py.get_param_nodes())
        msg = "expected {} params, but get {} params. ".format(
            num_param_inputs, len(params))
        msg += "This may due to your network have multi output. use param_exclude to remove them"
        assert len(params) == num_param_inputs, msg
        for pnode, param in zip(graph_py.get_param_nodes(), params):
            pnode.resolved_outputs[0] = param
    net = current_network()
    if net is not None:
        assert input_names is not None, "trt mode must provide input name"
        if not isinstance(input_names, (list, tuple)):
            input_names = [input_names]
        assert len(input_names) == len(example_inputs)
        inputs = []
        if input_tensors is not None:
            assert len(input_tensors) == len(example_inputs)
            inputs = input_tensors
        else:
            for torch_inp, name in zip(example_inputs, input_names):
                tensor = net.add_input(
                    name=name,
                    dtype=trt.float32,
                    shape=tuple(torch_inp.shape[1:]))
                inputs.append(tensor)
        for i, inode in enumerate(graph_py.get_input_nodes_dict().values()):
            inode.resolved_outputs[0] = inputs[i]
    else:
        # use torch inputs, debug mode
        for i, inode in enumerate(graph_py.get_input_nodes_dict().values()):
            inode.resolved_outputs[0] = example_inputs[i]
    resolve_graph(graph_py, output_names, verbose=verbose)
    # trace must be returned to avoid std::bad_alloc
    return trace, graph_py


def clean_resolved_outputs(graph_py, output_name):
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


def debug_call_graph(graph_py, inputs, output_names=None):
    assert current_network() is None, "must run in pytorch debug mode"
    if output_names is None:
        output_names = graph_py.get_output_names()
    if not isinstance(inputs, (list, tuple)):
        inputs = [inputs]
    if not isinstance(output_names, (list, tuple)):
        output_names = [output_names]
    for output_name in output_names:
        clean_resolved_outputs(graph_py, output_name)
    for i, inode in enumerate(graph_py.get_input_nodes_dict().values()):
        inode.resolved_outputs[0] = inputs[i]
    return resolve_graph(graph_py, output_names)
