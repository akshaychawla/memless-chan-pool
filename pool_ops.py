import torch 
from torch import nn 
from torch.utils.dlpack import to_dlpack, from_dlpack
import cupy as cp
import numpy as np 
import math, os, sys, time

def torch2cp(torch_tens):
    return cp.fromDlpack(to_dlpack(torch_tens))

def cp2torch(cp_tens):
    return from_dlpack(cp_tens.toDlpack())


class _FusedMultiPool(torch.autograd.Function): 
    
    @staticmethod 
    def forward(ctx, TORCH_input, TORCH_channel_idx_sets, d_out, GRAD_d_in, max_channels, kernels, DIMS):

        # continuous data checks 
        assert TORCH_input.is_contiguous(), "TORCH_input is not contiguous"
        assert TORCH_channel_idx_sets.is_contiguous(), "TORCH_channel_idx_sets is not contiguous"

        # torchTensor --> cp array
        d_in = torch2cp(TORCH_input)
        channel_idx_sets = torch2cp(TORCH_channel_idx_sets)

        # kernel parameters
        BATCHSIZE, CHANNELS, HEIGHT, WIDTH = DIMS["d_in_DIMS_list"]
        NUM_CHANNEL_SETS, MAX_CHANNELS_PER_SET = DIMS["channel_idx_sets_DIMS_list"]
        MAX_TILE_DIM = 8 
        NUM_TILES_X = math.ceil(float(WIDTH)/MAX_TILE_DIM)
        NUM_TILES_Y = math.ceil(float(HEIGHT)/MAX_TILE_DIM)
        # gridDims = (NUM_TILES_X, NUM_TILES_Y, BATCHSIZE)
        # blockDims = (MAX_TILE_DIM,MAX_TILE_DIM,NUM_CHANNEL_SETS) 
        gridDims = (NUM_TILES_X, NUM_TILES_Y, NUM_CHANNEL_SETS)
        blockDims = (MAX_TILE_DIM,MAX_TILE_DIM, BATCHSIZE) 

        # launch kernel 
        kernels["max_kernel_forward"](
            gridDims, blockDims, (d_out, DIMS["d_out_DIMS"], d_in, DIMS["d_in_DIMS"], channel_idx_sets, 
            DIMS["channel_idx_sets_DIMS"], max_channels, DIMS["max_channels_DIMS"], MAX_CHANNELS_PER_SET)
        )

        ctx.max_channels = max_channels
        ctx.kernels = kernels
        ctx.DIMS = DIMS
        ctx.GRAD_d_in = GRAD_d_in

        return cp2torch(d_out)

    @staticmethod
    def backward(ctx, TORCH_GRAD_d_out):

        # cheks if tensor is stored contigous manner in memory (row-major)
        assert TORCH_GRAD_d_out.is_contiguous(), "TORCH_GRAD_d_out is not contiguous"

        DIMS = ctx.DIMS
        # GRAD_d_in = ctx.GRAD_d_in
        GRAD_d_in = cp.zeros(DIMS["d_in_DIMS_list"], dtype=cp.float32)
        max_channels = ctx.max_channels

        # torch tensor --> cp array
        GRAD_d_out = torch2cp(TORCH_GRAD_d_out)

        # kernel parameters
        BATCHSIZE, CHANNELS, HEIGHT, WIDTH = DIMS["d_in_DIMS_list"]
        NUM_CHANNEL_SETS, MAX_CHANNELS_PER_SET = DIMS["channel_idx_sets_DIMS_list"]
        MAX_TILE_DIM = 8 
        NUM_TILES_X = math.ceil(float(WIDTH)/MAX_TILE_DIM)
        NUM_TILES_Y = math.ceil(float(HEIGHT)/MAX_TILE_DIM)
        gridDims = (NUM_TILES_X, NUM_TILES_Y, NUM_CHANNEL_SETS)
        blockDims = (MAX_TILE_DIM,MAX_TILE_DIM, BATCHSIZE) 

        # launch kernel 
        ctx.kernels["max_kernel_backward"](
            gridDims, blockDims, 
            (GRAD_d_out, DIMS["d_out_DIMS"], GRAD_d_in, DIMS["d_in_DIMS"], 
            max_channels, DIMS["max_channels_DIMS"])
        )

        return cp2torch(GRAD_d_in), None, None, None, None, None, None

class FusedMultiPool(nn.Module): 
    def __init__(self, channel_idx_sets): 
        super(FusedMultiPool, self).__init__()

        self.DIMS = None # check fxn precache 

        # compile the CUDA kernel
        with open("./kernel_test.cu", "rt") as f: 
            code = f.read() 
        max_kernel_forward = cp.RawKernel(code, "KERNEL_max_multi_FORWARD") # memoized. 
        max_kernel_backward = cp.RawKernel(code, "KERNEL_max_multi_BACKWARD")
        self.kernels = {
            "max_kernel_forward":max_kernel_forward,
            "max_kernel_backward":max_kernel_backward
            }

        # save channel_idx_sets (permanent tensor)
        self.channel_idx_sets = channel_idx_sets

    def precache(self, input_tensor_shape): 
        """
        Precache the following tensors/information: 
        d_in_DIMS, channel_idx_sets_DIMS, d_out_DIMS, d_out_DIMS_list, 
        max_channels_DIMS
        Also allocate memory for the following tensors: 
        self.d_out, self.max_channels
        This memory is pre-allocated and utilised across forward passes. 
        So long as the input shape is unchanged. 
        """

        print("[FusedPool] pre-caching..")
        BATCHSIZE, NUM_INPUT_CHANNELS, HEIGHT, WIDTH = input_tensor_shape 
        d_in_DIMS = cp.array([BATCHSIZE, NUM_INPUT_CHANNELS, HEIGHT, WIDTH], dtype=cp.int32)

        NUM_CHANNEL_SETS = self.channel_idx_sets.shape[0]
        MAX_CHANNELS_PER_SET = self.channel_idx_sets.shape[1]

        channel_idx_sets_DIMS = cp.array([NUM_CHANNEL_SETS, MAX_CHANNELS_PER_SET], dtype=cp.int32)
        d_out_DIMS = cp.array([BATCHSIZE, NUM_CHANNEL_SETS, HEIGHT, WIDTH], dtype=cp.int32)
        max_channels_DIMS = cp.array([BATCHSIZE, NUM_CHANNEL_SETS, HEIGHT, WIDTH], dtype=cp.int32)

        self.d_out = cp.zeros(d_out_DIMS.tolist(), dtype=cp.float32)
        self.max_channels = cp.ones(shape=max_channels_DIMS.tolist(), dtype=cp.int32)
        self.GRAD_d_in = cp.zeros(d_in_DIMS.tolist(), dtype=cp.float32)

        DIMS = {    
            "input_tensor_shape" : input_tensor_shape,
            "d_in_DIMS":d_in_DIMS, 
            "d_in_DIMS_list": cp.asnumpy(d_in_DIMS).tolist(),
            "channel_idx_sets_DIMS":channel_idx_sets_DIMS, 
            "channel_idx_sets_DIMS_list":cp.asnumpy(channel_idx_sets_DIMS).tolist(), 
            "d_out_DIMS":d_out_DIMS, 
            "d_out_DIMS_list": cp.asnumpy(d_out_DIMS).tolist(),
            "max_channels_DIMS":max_channels_DIMS,
            "max_channels_DIMS_list":cp.asnumpy(max_channels_DIMS).tolist()
        }
        return DIMS

    def forward(self, input): 

        # re-cache dims and persistant tensors 
        if self.DIMS == None: 
            self.DIMS = self.precache(input.shape)
        elif tuple(input.shape) != tuple(self.DIMS["input_tensor_shape"]):
            self.DIMS = self.precache(input.shape)

        return _FusedMultiPool.apply(
                input, self.channel_idx_sets, self.d_out, self.GRAD_d_in, 
                self.max_channels, self.kernels, self.DIMS)



