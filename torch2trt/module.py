import numpy as np
import tensorrt as trt
import torch
from torch import nn

import torch2trt
from torch2trt.inference.inference import TorchInferenceContext
from torch2trt.utils import get_torch_forward_name
from torch.utils import dlpack
try:
    import tvm 
    from tvm.relay import expr, analysis
    from tvm import relay
    from tvm.contrib import graph_runtime
except ImportError:
    pass

class TensorRTModule(nn.Module):
    def __init__(self,
                 max_batchsize,
                 workspace,
                 dtype=trt.float32,
                 builder_config_fn=None,
                 net_post_fn=None,
                 input_names=None,
                 verbose=False):
        super().__init__()
        self.max_batchsize = max_batchsize
        self.workspace = workspace
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.built = False
        self.graph_pth = None
        self.refit_weight_dict = {}
        self.engine = None
        self.ctx = None
        self.output_shapes = None
        self.output_names = None
        self.need_refit = False
        self.verbose = verbose
        self.builder_config_fn = builder_config_fn
        self.input_names = input_names
        self.net_post_fn = net_post_fn

    def build_tensorrt(self, net, torch_inputs):
        if self.input_names is None:
            input_names = get_torch_forward_name(net.forward)
        else:
            input_names = self.input_names
        self.graph_pth = torch2trt.GraphModule(net, torch_inputs)
        self.output_names = []
        with trt.Builder(
                self.logger) as builder, builder.create_network() as trt_net:
            builder.max_workspace_size = self.workspace
            builder.max_batch_size = self.max_batchsize
            builder.fp16_mode = builder.platform_has_fast_fp16
            # builder.refittable = False
            if self.builder_config_fn is not None:
                self.builder_config_fn(builder)
            with torch2trt.trt_network(trt_net):
                inputs = []
                for i, arg in enumerate(torch_inputs):
                    name = input_names[i]
                    inp = trt_net.add_input(name=name,
                                            shape=arg.shape[1:],
                                            dtype=trt.float32)
                    inputs.append(inp)
                outputs = self.graph_pth(*inputs, verbose=self.verbose)
            self.refit_weight_dict = self.graph_pth.graph.refit_weight_dict
            if not isinstance(outputs, (list, tuple)):
                outputs = [outputs]
            for i, out in enumerate(outputs):
                name = "output{}".format(i)
                out.name = name
                self.output_names.append(name)
                trt_net.mark_output(tensor=out)
            self.builder = builder
            if self.net_post_fn is not None:
                self.net_post_fn(trt_net)
            self.engine = builder.build_cuda_engine(trt_net)
            self.ctx = self.engine.create_execution_context()
            self.ctx = torch2trt.TorchInferenceContext(self.ctx)
        # get output shapes
        outputs = self.graph_pth(*torch_inputs)
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
        self.output_shapes = {}
        for n, v in zip(self.output_names, outputs):
            self.output_shapes[n] = v.shape[1:]

    def refit_engine(self, net):
        print(
            "TensorRT refit seems not working with batchnorm. so disable it for now."
        )
        raise NotImplementedError
        variables = []
        with trt.Refitter(self.engine, self.logger) as refitter:
            net = net.eval()
            state_dict = self.graph_pth.collect_params(net)
            # Why use a variable list?
            # we know that in c++ functions, a python array may be deleted
            # after ref count of a var decrease to zero.
            # TensorRT 5.1.5.0 refitter ONLY EXECUTED in refit_cuda_engine,
            # so we must keep variable alive before refit_cuda_engine call.
            for k, v in self.refit_weight_dict.items():
                if v["type"] == "Linear":
                    weight = state_dict[v["weight"]].detach().cpu().numpy()
                    refitter.set_weights(k, trt.WeightsRole.KERNEL, weight)
                    variables.append(weight)
                    if "bias" in v:
                        bias = state_dict[v["bias"]].detach().cpu().numpy()
                        refitter.set_weights(k, trt.WeightsRole.BIAS, bias)
                        variables.append(bias)
                elif v["type"] == "Convolution":
                    weight = state_dict[
                        v["weight"]].detach().float().cpu().numpy()
                    refitter.set_weights(k, trt.WeightsRole.KERNEL, weight)
                    variables.append(weight)
                    if "bias" in v:
                        bias = state_dict[v["bias"]].detach().cpu().numpy()
                        refitter.set_weights(k, trt.WeightsRole.BIAS, bias)
                        variables.append(bias)
                elif v["type"] == "BatchNorm":
                    running_var = state_dict[v["running_var"]]
                    running_mean = state_dict[v["running_mean"]]
                    weight = state_dict[v["weight"]]
                    bias = state_dict[v["bias"]]
                    eps = v["eps"]
                    running_mean = running_mean.detach().cpu().numpy()
                    running_var = running_var.detach().cpu().numpy()
                    weight = weight.detach().cpu().numpy()
                    bias = bias.detach().cpu().numpy()
                    shift = (-running_mean /
                             np.sqrt(running_var + eps)) * weight + bias
                    scale = weight / np.sqrt(running_var + eps)
                    shift = np.ascontiguousarray(shift.astype(np.float32))
                    scale = np.ascontiguousarray(scale.astype(np.float32))
                    refitter.set_weights(k, trt.WeightsRole.SHIFT, shift)
                    refitter.set_weights(k, trt.WeightsRole.SCALE, scale)
                    variables.append(scale)
                    variables.append(shift)
                else:
                    raise NotImplementedError
            # Get description of missing weights. This should return empty
            # lists in this case.
            [missingLayers, weightRoles] = refitter.get_missing()
            assert len(
                missingLayers
            ) == 0, "Refitter found missing weights. Call set_weights() for all missing weights"
            # Refit the engine with the new weights. This will return True if
            # the refit operation succeeded.
            assert refitter.refit_cuda_engine()

    def __call__(self, *args, **kw):
        if not self.training and not self.built:
            self.build_tensorrt(self, args)
            self.built = True
            self.need_refit = False
        if not self.training:
            if self.need_refit:
                self.refit_engine(self)
                self.need_refit = False
            assert all([a.is_cuda for a in args])
            torch.cuda.synchronize()
            output_dict = self.ctx.inference_async(*args)
            outputs = [None] * len(output_dict)
            for k, v in output_dict.items():
                outputs[self.output_names.index(k)] = v.view(
                    v.shape[0], *self.output_shapes[k])
            if len(outputs) == 1:
                return outputs[0]
            return tuple(outputs)
        else:
            self.need_refit = True
            return super().__call__(*args, **kw)


class TensorRTModuleWrapper(TensorRTModule):
    def __init__(self,
                 net,
                 max_batchsize,
                 workspace,
                 dtype=trt.float32,
                 builder_config_fn=None,
                 net_post_fn=None,
                 input_names=None,
                 verbose=False):
        super().__init__(max_batchsize, workspace, dtype, builder_config_fn,
                         net_post_fn, input_names, verbose)
        self.net = net

    def forward(self, *args, **kw):
        return self.net.forward(*args, **kw)

    def __call__(self, *args, **kw):
        if not self.training and not self.built:
            self.build_tensorrt(self.net, args)
            self.built = True
            self.need_refit = False
        if not self.training:
            if self.need_refit:
                self.refit_engine(self.net)
                self.need_refit = False
            assert all([a.is_cuda for a in args])
            torch.cuda.synchronize()
            # args = [a.detach().cpu().numpy() for a in args]
            output_dict = self.ctx.inference_async(*args)
            # for k,v in output_dict.items():
            #     output_dict[k] = torch.tensor(v, dtype=torch.float32, device=torch.device("cuda:0"))
            outputs = [None] * len(output_dict)
            for k, v in output_dict.items():
                outputs[self.output_names.index(k)] = v.view(
                    v.shape[0], *self.output_shapes[k])
            if len(outputs) == 1:
                return outputs[0]
            return tuple(outputs)
        else:
            self.need_refit = True
            return super().__call__(*args, **kw)


class TVMInference:
    def __init__(self, mod, params, ctx=None, input_names=None, cudnn=False, opt_level=3):
        target = "cuda"
        if cudnn:
            target += " -libs=cudnn"
        # target = 'llvm -mcpu=core-avx2'
        with relay.build_config(opt_level=3):
            graph, lib, params = relay.build(mod, target, params=params)
        self.graph = graph
        self.lib = lib
        self.ctx = ctx or tvm.gpu(0)
        # self.ctx = ctx or tvm.cpu()
        self.tvm_context = graph_runtime.create(graph, lib, self.ctx)
        self.params = params
        self.tvm_context.set_input(**params)
        self.input_names = input_names

    def update_params(self, params):
        self.params = params
        self.tvm_context.set_input(**params)

    def __call__(self, *args, **kw):
        return self.inference(*args, **kw)

    def inference(self, *args, **kw):
        if self.input_names is None:
            assert len(args) == 0 and len(kw) != 0
            for k, v in kw.items():
                self.tvm_context.set_input(k, tvm.nd.array(v, ctx=self.ctx))
        else:
            assert len(args) == len(self.input_names)
            for k, v in zip(self.input_names, args):
                self.tvm_context.set_input(k, tvm.nd.array(v, ctx=self.ctx))
        self.tvm_context.run()
        num_outputs = self.tvm_context.get_num_outputs()
        outputs = []
        for i in range(num_outputs):
            outputs.append(self.tvm_context.get_output(i).asnumpy())
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    def inference_torch(self, *args, **kw):
        if self.input_names is None:
            assert len(args) == 0 and len(kw) != 0
            for k, v in kw.items():
                if isinstance(v, torch.Tensor):
                    self.tvm_context.set_input(k, tvm.nd.from_dlpack(dlpack.to_dlpack(v)))
                else:
                    self.tvm_context.set_input(k, tvm.nd.array(v, ctx=self.ctx))
        else:
            assert len(args) == len(self.input_names)
            for k, v in zip(self.input_names, args):
                if isinstance(v, torch.Tensor):
                    self.tvm_context.set_input(k, tvm.nd.from_dlpack(dlpack.to_dlpack(v)))
                else:
                    self.tvm_context.set_input(k, tvm.nd.array(v, ctx=self.ctx))
        self.tvm_context.run()
        num_outputs = self.tvm_context.get_num_outputs()
        outputs = []
        for i in range(num_outputs):
            # outputs = [dlpack.from_dlpack(o) for o in outputs]
            out = self.tvm_context.get_output(i).to_dlpack()
            outputs.append(dlpack.from_dlpack(out))
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)


class TVMModule(nn.Module):
    def __init__(self,
                 graph_post_fn=None,
                 input_names=None,
                 verbose=False):
        super().__init__()
        self.built = False
        self.graph_pth = None
        self.refit_weight_dict = {}
        self.ctx = None
        self.output_names = None
        # self.need_refit = True
        self.verbose = verbose
        self.input_names = input_names
        self.params = None
        self.graph_post_fn = graph_post_fn # can convert graph to quantized graph

    def build_tvm(self, net, torch_inputs):
        self.graph_pth = torch2trt.GraphModule(net, torch_inputs)
        with torch2trt.core.tvm_network():
            trace, graph_pth = torch2trt.core.torch2tvm(
                net,
                torch_inputs,
                input_names=self.input_names,
                verbose=self.verbose)
        self.refit_weight_dict = graph_pth.refit_weight_dict
        input_names = get_torch_forward_name(net.forward)
        self.graph_pth = graph_pth
        outputs = graph_pth.get_resolved_outputs()
        tvm_weight_dict = graph_pth.context.tvm_weight_dict
        self.params = {k.name_hint: v for k, v in tvm_weight_dict.items()}
        print(len(self.params))
        self.graph = expr.Function(analysis.free_vars(outputs), outputs)
        if self.graph_post_fn is not None:
            self.graph = self.graph_post_fn(self.graph)
        self.ctx = TVMInference(self.graph, self.params, input_names=input_names, ctx=tvm.gpu(0), cudnn=True)

    def refit_engine(self, net):
        input_names = get_torch_forward_name(net.forward)
        params = self.graph_pth.torch_weight_nodes_dict
        params = {n: params[v].detach().cpu().numpy() for n, v in self.refit_weight_dict.items()}
        self.params = params
        self.ctx = TVMInference(self.graph, params, input_names=input_names, ctx=tvm.gpu(0), cudnn=True)

    def __call__(self, *args, **kw):
        if not self.training and not self.built:
            self.build_tvm(self, args)
            self.built = True
            self.need_refit = False
        if not self.training:
            if self.need_refit:
                self.refit_engine(self)
                self.need_refit = False
            assert all([a.is_cuda for a in args])
            outputs = self.ctx.inference_torch(*args)
            if len(outputs) == 1:
                return outputs[0]
            return tuple(outputs)
        else:
            self.need_refit = True
            return super().__call__(*args, **kw)

class TVMModuleWrapper(TVMModule):
    def __init__(self,
                net,
                 graph_post_fn=None,
                 input_names=None,
                 verbose=False):
        super().__init__(graph_post_fn, input_names, verbose)
        self.net = net

    def forward(self, *args, **kw):
        return self.net.forward(*args, **kw)

    def __call__(self, *args, **kw):
        if not self.training and not self.built:
            self.build_tvm(self.net, args)
            self.built = True
            self.need_refit = False
        if not self.training:
            if self.need_refit:
                self.refit_engine(self.net)
                self.need_refit = False
            assert all([a.is_cuda for a in args])
            outputs = self.ctx.inference_torch(*args)
            if len(outputs) == 1:
                return outputs[0]
            return tuple(outputs)
        else:
            self.need_refit = True
            return super().__call__(*args, **kw)
