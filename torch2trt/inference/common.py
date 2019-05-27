#
# Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO LICENSEE:
#
# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
#
# These Licensed Deliverables contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a form of NVIDIA software license agreement by and
# between NVIDIA and Licensee ("License Agreement") or electronically
# accepted by Licensee.  Notwithstanding any terms or conditions to
# the contrary in the License Agreement, reproduction or disclosure
# of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
#
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.
#
# Any use of the Licensed Deliverables in individual and commercial
# software must include, in the user documentation and internal
# comments to the code, the above Disclaimer and U.S. Government End
# Users Notice.
#

import os
import argparse
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import tensorrt as trt

# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem, name, index):
        self.host = host_mem
        self.device = device_mem
        self.index = index
        self.name = name
        self.device_input = False

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()

# Allocates all buffers required for an engine, i.e. host/device inputs/outputs.
def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    index = 0
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        shape = [engine.max_batch_size] + list(engine.get_binding_shape(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype).reshape(shape)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem, binding, index))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem, binding, index))
        index += 1
    return inputs, outputs, bindings

def torch_to_np_dtype_map():
    import torch
    type_map = {
        torch.float16: np.dtype(np.float16),
        torch.float32: np.dtype(np.float32),
        torch.float64: np.dtype(np.float64),
        torch.int32: np.dtype(np.int32),
        torch.int64: np.dtype(np.int64),
        torch.uint8: np.dtype(np.uint8),
    }
    return type_map


def np_to_torch_dtype_map():
    type_map = torch_to_np_dtype_map()
    return {v: k for k, v in type_map.items()}


def allocate_buffers_torch(engine: trt.ICudaEngine, device):
    import torch
    inputs = []
    outputs = []
    bindings = []
    index = 0
    dtype_map = np_to_torch_dtype_map()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        shape = [engine.max_batch_size] + list(engine.get_binding_shape(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype).reshape(shape)
        device_mem = torch.empty(*host_mem.shape, device=device, dtype=dtype_map[host_mem.dtype])
        # Append the device buffer to device bindings.
        bindings.append(device_mem.data_ptr())
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem, binding, index))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem, binding, index))
        index += 1
    return inputs, outputs, bindings
