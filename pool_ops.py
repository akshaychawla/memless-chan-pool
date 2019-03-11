import torch 
from torch import nn 
from torch.utils.dlpack import to_dlpack, from_dlpack
import cupy as cp
import numpy as np 
import math, os, sys, time

class _FusedMultiPool(torch.autograd.Function): 
    
    @staticmethod 
    def forward(ctx, TORCH_input, TORCH_channel_idx_sets, max_kernel_forward, max_kernel_backward):

        # continuous data checks 
        assert TORCH_input.is_contiguous(), "TORCH_input is not contiguous"
        assert TORCH_channel_idx_sets.is_contiguous(), "TORCH_channel_idx_sets is not contiguous"


        d_in = cp.fromDlpack(to_dlpack(TORCH_input))
        d_in_DIMS = cp.array(d_in.shape, dtype=cp.int32)
        batchsize, CHANNELS, HEIGHT, WIDTH = d_in.shape

        channel_idx_sets = cp.fromDlpack(to_dlpack(TORCH_channel_idx_sets))
        channel_idx_sets_DIMS = cp.array(channel_idx_sets.shape, dtype=cp.int32)

        NUM_CHANNEL_SETS = channel_idx_sets.shape[0]
        MAX_CHANNELS_PER_SET = channel_idx_sets.shape[1]
        d_out_DIMS = cp.array([batchsize, NUM_CHANNEL_SETS, HEIGHT, WIDTH], dtype=cp.int32)
        d_out = cp.zeros(cp.asnumpy(d_out_DIMS).tolist(), dtype=cp.float32)

        MAX_TILE_DIM = 4 
        NUM_TILES_X = math.ceil(float(WIDTH)/MAX_TILE_DIM)
        NUM_TILES_Y = math.ceil(float(HEIGHT)/MAX_TILE_DIM)

        max_channels_DIMS = cp.array([batchsize, NUM_CHANNEL_SETS, HEIGHT, WIDTH], dtype=cp.int32)
        max_channels = cp.ones(shape=max_channels_DIMS.tolist(), dtype=cp.int32)

        gridDims = (NUM_TILES_X, NUM_TILES_Y, batchsize)
        blockDims = (MAX_TILE_DIM,MAX_TILE_DIM,NUM_CHANNEL_SETS) 

        # run the kernel 
        max_kernel_forward(
            gridDims, blockDims, (d_out, d_out_DIMS, d_in, d_in_DIMS, channel_idx_sets, 
            channel_idx_sets_DIMS, max_channels, max_channels_DIMS, MAX_CHANNELS_PER_SET)
        )

        ctx.max_channels = max_channels
        ctx.max_channels_DIMS = max_channels_DIMS
        ctx.d_in_DIMS = d_in_DIMS
        ctx.d_out_DIMS = d_out_DIMS
        ctx.max_kernel_forward = max_kernel_forward
        ctx.max_kernel_backward = max_kernel_backward

        return from_dlpack(d_out.toDlpack())

    @staticmethod
    def backward(ctx, TORCH_GRAD_d_out):

        # cheks if tensor is stored contigous manner in memory (row-major)
        assert TORCH_GRAD_d_out.is_contiguous(), "TORCH_GRAD_d_out is not contiguous"

        max_channels = ctx.max_channels
        max_channels_DIMS = ctx.max_channels_DIMS
        d_in_DIMS = ctx.d_in_DIMS 
        GRAD_d_in_DIMS = ctx.d_in_DIMS
        GRAD_d_out_DIMS = ctx.d_out_DIMS
        GRAD_d_in = cp.zeros(shape=GRAD_d_in_DIMS.tolist(), dtype=cp.float32)
        GRAD_d_out = cp.fromDlpack(to_dlpack(TORCH_GRAD_d_out))
        NUM_CHANNEL_SETS = GRAD_d_in_DIMS.asnumpy()[1]

        batchsize = GRAD_d_in_DIMS.asnumpy()[0]
        MAX_TILE_DIM = 4 
        NUM_TILES_X = math.ceil(float(WIDTH)/MAX_TILE_DIM)
        NUM_TILES_Y = math.ceil(float(HEIGHT)/MAX_TILE_DIM)

        gridDims = (NUM_TILES_X, NUM_TILES_Y, batchsize)
        blockDims = (MAX_TILE_DIM,MAX_TILE_DIM,NUM_CHANNEL_SETS) 

        self.max_kernel_backward(
            gridDims, blockDims,
            (GRAD_d_out, GRAD_d_out_DIMS, GRAD_d_in, GRAD_d_in_DIMS, max_channels, max_channels_DIMS)
        )

        return GRAD_d_in, None, None, None

class FusedMultiPool(nn.Module): 
    def __init__(self, channel_idx_sets): 
        super(FusedMultiPool, self).__init__()

        # compile the CUDA kernel
        with open("./kernel_test.cu", "rt") as f: 
            code = f.read() 
        self.max_kernel_forward = cp.RawKernel(code, "KERNEL_max_multi_FORWARD") # memoized. 
        self.max_kernel_backward = cp.RawKernel(code, "KERNEL_max_multi_BACKWARD")

        # save channel_idx_sets (permanent tensor)
        self.channel_idx_sets = channel_idx_sets
    
    def forward(self, input): 
        return _FusedMultiPool.apply(input, self.channel_idx_sets, self.max_kernel_forward, self.max_kernel_backward)

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

def builtin_chan_pool(x, channel_idx_sets):
    channel_idx_sets = channel_idx_sets.long()
    y = torch.zeros(16, 40, 64, 128, dtype=torch.float32).cuda()
    for i in range(40):
        new_tens = torch.index_select(x, 1, channel_idx_sets[i])
        y_single_indcs,_ = torch.max(new_tens, 1) 
        y[:, i] += y_single_indcs
    return y


def benchmark(): 
    batchsize = 16 
    x = torch.randn(batchsize, 256, 64, 128).cuda()

    # generate channel_idx_sets 
    channel_idx_sets = [] 
    num_sets = 40 
    max_channels_per_set = 102 # each set covers 40% of the input channels
    for num_set in range(num_sets): 
        channel_idx = np.random.randint(0, 256, size=(max_channels_per_set,))
        channel_idx = np.sort(channel_idx) 
        channel_idx_sets.append(channel_idx)
    channel_idx_sets = np.array(channel_idx_sets).astype(np.int32)
    
    # benchmark FusedMultiPool
    channel_idx_sets = torch.from_numpy(channel_idx_sets).cuda() 
    test_module = FusedMultiPool(channel_idx_sets) 
    time_durations = []
    for _ in range(100): 
        start = time.time()
        y = test_module.forward(x) 
        time_durations.append(time.time() - start)
        del y
    print("Custom kernel took on avg ", np.mean(time_durations[1:]))

    # benchmark standard multi channel pooling 
    time_durations = []
    for _ in range(100):
        start = time.time() 
        y = builtin_chan_pool(x, channel_idx_sets)
        time_durations.append(time.time() - start)
        del y
    print("Regular ol' pytorch took on avg ", np.mean(time_durations[1:]))
    import ipdb; ipdb.set_trace()

    


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
    benchmark()
    # test()
