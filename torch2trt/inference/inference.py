import tensorrt as trt
import numpy as np
from torch2trt.inference.common import allocate_buffers, allocate_buffers_torch, np_to_torch_dtype_map
import contextlib
import pycuda.driver as cuda
import pycuda
import torch


class InferenceContext:
    def __init__(self, context: trt.IExecutionContext, stream=None, cuda_device=None, cuda_context=None):
        self.engine = context.engine
        inputs, outputs, bindings = allocate_buffers(self.engine)
        self.context = context
        self.inputs = inputs
        self.outputs = outputs
        self.bindings = bindings
        self.input_dict = {mem.name: mem for mem in inputs}
        self.output_dict = {mem.name: mem for mem in outputs}
        if stream is None:
            self.stream = cuda.Stream()
        self._batch_size = None
        self.cuda_device = cuda_device
        self.cuda_context = cuda_context

    @property
    def max_batch_size(self):
        return self.engine.max_batch_size

    @contextlib.contextmanager
    def device(self):
        if self.cuda_context is not None:
            self.cuda_context.push()
            yield 
            self.cuda_context.pop()
        else:
            yield


    @contextlib.contextmanager
    def inference_io(self, *inputs, **kwargs):
        """get batch size, prepare inputs
        """
        assert len(inputs) == 0 or len(kwargs) == 0
        batch_sizes = []
        inputs = list(inputs)
        if len(inputs) == 0:
            assert len(kwargs) == len(self.inputs)
            inputs = [kwargs[mem.name] for mem in self.inputs]
        else:
            assert len(inputs) == len(self.inputs)
        for v, mem in zip(inputs, self.inputs):
            size = mem.host.size // self.max_batch_size
            if isinstance(v, torch.Tensor):
                v_size = v.numel()
            else:
                v_size = v.size
            assert v_size % size == 0
            batch_sizes.append(v_size // size)
        # process torch input
        bindings_new = [b for b in self.bindings]
        bindings_backup = [b for b in self.bindings]
        host_backup = [mem.host for mem in self.inputs]
        for i in range(len(inputs)):
            if isinstance(inputs[i], torch.Tensor):
                # TODO: currently we can't ensure device of engine is same as torch device.
                if inputs[i].is_cuda:
                    msg = "only support inference in device 0 for now if you use torch cuda input"
                    assert inputs[i].device == torch.device("cuda:0"), msg
                    bindings_new[i] = int(inputs[i].data_ptr())
                    self.inputs[i].device_input = True
                else:
                    inputs[i] = inputs[i].detach().numpy()
                    self.inputs[i].device_input = False
            else:
                self.inputs[i].device_input = False

        assert all(b == batch_sizes[0] for b in batch_sizes)
        batch_size = batch_sizes[0]
        assert batch_size <= self.max_batch_size, "your batch size is too large."
        
        for i in range(len(inputs)):
            if not self.inputs[i].device_input:
                self.inputs[i].host[:batch_size] = inputs[i]
        
        self.bindings = bindings_new
        self._batch_size = batch_size
        yield
        for mem, inp in zip(self.inputs, host_backup):
            mem.host = inp
        self._batch_size = None
        self.bindings = bindings_backup

    def execute_async(self, batch_size):
        [
            cuda.memcpy_htod_async(inp.device, inp.host[:batch_size],
                                   self.stream) for inp in self.inputs
            if inp.device_input is False
        ]
        self.context.execute_async(
            batch_size=batch_size,
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        [
            cuda.memcpy_dtoh_async(out.host[:batch_size], out.device,
                                   self.stream) for out in self.outputs
        ]
        self.stream.synchronize()
        return {n: v.host[:batch_size] for n, v in self.output_dict.items()}

    def execute(self, batch_size):
        [
            cuda.memcpy_htod(inp.device, inp.host[:batch_size])
            for inp in self.inputs if inp.device_input is False
        ]
        self.context.execute(batch_size=batch_size, bindings=self.bindings)
        [
            cuda.memcpy_dtoh(out.host[:batch_size], out.device)
            for out in self.outputs
        ]
        return {n: v.host[:batch_size] for n, v in self.output_dict.items()}

    def inference(self, *inputs, **kwargs):
        """do inference sync. should only be used when profiling.
        Args: 
            *inputs: list of numpy array
            **kwargs: dict of name to array
        Returns:
            outputs: dict of name to output array
        """
        with self.device():
            with self.inference_io(*inputs, **kwargs):
                assert self._batch_size is not None
                return self.execute(self._batch_size)

    def inference_async(self, *inputs, **kwargs):
        """do inference.
        Args: 
            *inputs: list of numpy array
            **kwargs: dict of name to array
        Returns:
            outputs: dict of name to output array
        """
        with self.device():
            with self.inference_io(*inputs, **kwargs):
                assert self._batch_size is not None
                return self.execute_async(self._batch_size)


class TorchInferenceContext(InferenceContext):
    """same as InferenceContext except this class always take and return torch cuda tensor.
    """
    def __init__(self, context: trt.IExecutionContext, stream=None, device=None, cuda_device=None, cuda_context=None):
        self.engine = context.engine
        if device is None:
            self.torch_device = torch.device("cuda:0")
        else:
            self.torch_device = device
        inputs, outputs, bindings = allocate_buffers_torch(self.engine, self.torch_device)
        self.context = context
        self.inputs = inputs
        self.outputs = outputs
        self.bindings = bindings
        self.input_dict = {mem.name: mem for mem in inputs}
        self.output_dict = {mem.name: mem for mem in outputs}
        if stream is None:
            self.stream = cuda.Stream()
        self._batch_size = None
        self.cuda_device = cuda_device
        self.cuda_context = cuda_context

    def execute_async(self, batch_size):
        assert all([inp.device_input for inp in self.inputs]), "all input must be cuda tensor"
        for i in range(len(self.inputs)):
            inp = self.inputs[i]
            if inp.device_input is False:
                cuda.memcpy_htod_async(self.bindings[i], inp.host[:batch_size], self.stream)
        self.context.execute_async(
            batch_size=batch_size,
            bindings=self.bindings,
            stream_handle=self.stream.handle)
        # must sync here because we use custom stream instead of torch stream
        self.stream.synchronize()
        return {n: v.device[:batch_size] for n, v in self.output_dict.items()}

    def execute(self, batch_size):
        """
        for i in range(len(self.inputs)):
            inp = self.inputs[i]
            if inp.device_input is False:
                cuda.memcpy_htod(self.bindings[i], inp.host[:batch_size])
        self.context.execute(batch_size=batch_size, bindings=self.bindings)
        return {n: v.device[:batch_size] for n, v in self.output_dict.items()}
        """
        raise NotImplementedError
