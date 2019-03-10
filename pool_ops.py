import torch 
from torch import nn 
from torch.utils.dlpack import to_dlpack, from_dlpack
import cupy as cp
import numpy as np 
import math, os, sys

class _FusedMultiPool(torch.autograd.Function): 
    
    @staticmethod 
    def forward(ctx, TORCH_input, TORCH_channel_idx_sets, max_kernel):

        d_in = cp.fromDlpack(to_dlpack(TORCH_input))
        d_in_DIMS = cp.array(d_in.shape, dtype=cp.int32)
        batchsize, CHANNELS, HEIGHT, WIDTH = d_in.shape

        channel_idx_sets = cp.fromDlpack(to_dlpack(TORCH_channel_idx_sets))
        channel_idx_sets_DIMS = cp.array(channel_idx_sets.shape, dtype=cp.int32)

        NUM_CHANNEL_SETS = channel_idx_sets.shape[0]
        MAX_CHANNELS_PER_SET = channel_idx_sets.shape[1]
        d_out_DIMS = cp.array([batchsize, NUM_CHANNEL_SETS, HEIGHT, WIDTH], dtype=cp.int32)
        d_out = cp.zeros(cp.asnumpy(d_out_DIMS).tolist(), dtype=cp.float32)

        max_channels_DIMS = cp.array([batchsize, NUM_CHANNEL_SETS, HEIGHT, WIDTH], dtype=cp.int32)
        max_channels = cp.ones(shape=max_channels_DIMS.tolist(), dtype=cp.int32)

        MAX_TILE_DIM = 4 
        NUM_TILES_X = math.ceil(float(WIDTH)/MAX_TILE_DIM)
        NUM_TILES_Y = math.ceil(float(HEIGHT)/MAX_TILE_DIM)

        max_channels_DIMS = cp.array([batchsize, NUM_CHANNEL_SETS, HEIGHT, WIDTH], dtype=cp.int32)
        max_channels = cp.ones(shape=max_channels_DIMS.tolist(), dtype=cp.int32)
        import ipdb; ipdb.set_trace()

        gridDims = (NUM_TILES_X, NUM_TILES_Y, batchsize)
        blockDims = (MAX_TILE_DIM,MAX_TILE_DIM,NUM_CHANNEL_SETS) 

        # run the kernel 
        max_kernel(
            gridDims, blockDims, (d_out, d_out_DIMS, d_in, d_in_DIMS, channel_idx_sets, 
            channel_idx_sets_DIMS, max_channels, max_channels_DIMS, MAX_CHANNELS_PER_SET)
        )

        return from_dlpack(d_out.toDlpack())

    @staticmethod
    def backward(ctx, grad_output):
        return None, None

class FusedMultiPool(nn.Module): 
    def __init__(self, channel_idx_sets): 
        super(FusedMultiPool, self).__init__()

        # compile the CUDA kernel
        with open("./kernel_test.cu", "rt") as f: 
            code = f.read() 
        self.max_kernel = cp.RawKernel(code, "KERNEL_max_multi") # memoized. 

        # save channel_idx_sets (permanent tensor)
        self.channel_idx_sets = channel_idx_sets
    
    def forward(self, input): 
        return _FusedMultiPool.apply(input, self.channel_idx_sets, self.max_kernel)

def multi_pool_numpy(d_in, channel_idx_sets):
    d_in_cp = cp.fromDlpack(to_dlpack(d_in))
    channel_idx_sets_cp = cp.fromDlpack(to_dlpack(channel_idx_sets))
    num_sets, MAX_CHANNELS_PER_SET = channel_idx_sets_cp.shape
    _,rows,cols = d_in_cp.shape
    d_in_cp_cpu = cp.asnumpy(d_in_cp)
    d_out_cpu = np.zeros(shape=(num_sets,rows, cols))
    channel_idx_sets_cp_cpu = cp.asnumpy(channel_idx_sets_cp)
    for i in range(num_sets):
        chan_set = channel_idx_sets_cp_cpu[i] 
        d_in_cp_sub = d_in_cp_cpu[chan_set, :, :]
        op = np.amax(d_in_cp_sub, axis=0)
        d_out_cpu[i] += op
    
    return d_out_cpu

def test(): 

    # torch is (batch, channels, height, width) perfect.
    batchsize = 5
    x = torch.randn(batchsize, 4, 64, 128).cuda()
    channel_idx_sets = torch.tensor([[0,1],[0,3],[1,2]], dtype=torch.int32).cuda() 
    test_module = FusedMultiPool(channel_idx_sets)
    y = test_module.forward(x)

    # check if numpy output is same as CUDA output 
    for batch_idx in range(batchsize):
        print("batch_idx: ", batch_idx)
        op = multi_pool_numpy(x[batch_idx], channel_idx_sets)
        print("Same?", np.allclose(y.cpu().numpy()[batch_idx], op))

if __name__ == "__main__":
    test()
