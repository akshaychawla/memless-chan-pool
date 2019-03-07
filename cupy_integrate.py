import numpy as np 
import cupy as cp 
import cupy
import math

def verify_output(d_out, d_in):
    d_out_cpu = cp.asnumpy(d_out)
    d_in_cpu = cp.asnumpy(d_in)
    chanpool = np.amax(d_in_cpu, axis=0)
    print("Same? {}".format(np.allclose(chanpool, d_out_cpu)))

def multi_pool_numpy(d_in, channel_idx_sets):
    num_sets, MAX_CHANNELS_PER_SET = channel_idx_sets.shape
    _,rows,cols = d_in.shape
    d_in_cpu = cp.asnumpy(d_in)
    d_out_cpu = np.zeros(shape=(num_sets,rows, cols))
    channel_idx_sets_cpu = cp.asnumpy(channel_idx_sets)
    for i in range(num_sets):
        chan_set = channel_idx_sets_cpu[i] 
        d_in_sub = d_in_cpu[chan_set, :, :]
        op = np.amax(d_in_sub, axis=0)
        d_out_cpu[i] += op
    
    return d_out_cpu



with open("./kernel_test.cu", "rt") as f: 
    code = f.read() 


DEPTH = 4 
HEIGHT = 2 
WIDTH = 3
batchsize = 1

max_kernel = cp.RawKernel(code, "KERNEL_max_multi")

MAX_CHANNELS_PER_SET = 2
NUM_CHANNEL_SETS = 3

# allocate tensors 
d_in = cp.random.randn(batchsize, DEPTH, HEIGHT,WIDTH, dtype=cp.float32) 
d_in_DIMS = cp.array([batchsize, DEPTH, HEIGHT, WIDTH], dtype=cp.int32)

d_out_DIMS = cp.array([batchsize, NUM_CHANNEL_SETS,HEIGHT, WIDTH], dtype=cp.int32)
d_out = cp.zeros(shape=d_out_DIMS.tolist(), dtype=cp.float32)

max_channels_DIMS = cp.array([batchsize, NUM_CHANNEL_SETS, HEIGHT, WIDTH], dtype=cp.int32)
max_channels = cp.ones(shape=max_channels_DIMS.tolist(), dtype=cp.int32)

channel_idx_sets_DIMS = cp.array([3,2], dtype=cp.int32)
channel_idx_sets = cp.array([ [0,1], [0,3] , [1,2] ], dtype=cp.int32) 

MAX_TILE_DIM = 4 
NUM_TILES_X = math.ceil(float(WIDTH)/MAX_TILE_DIM)
NUM_TILES_Y = math.ceil(float(HEIGHT)/MAX_TILE_DIM)


grid = (NUM_TILES_X, NUM_TILES_Y, batchsize)
block = (MAX_TILE_DIM,MAX_TILE_DIM,NUM_CHANNEL_SETS) 

# run the kernel 
max_kernel(grid, block, (d_out, d_out_DIMS, d_in, d_in_DIMS, channel_idx_sets, channel_idx_sets_DIMS, max_channels, max_channels_DIMS, MAX_CHANNELS_PER_SET))

# check if numpy output is same as CUDA output 
for batch_idx in range(batchsize):
    print("batch_idx: ", batch_idx)
    op = multi_pool_numpy(d_in[batch_idx], channel_idx_sets)
    print("Same?", np.allclose(cp.asnumpy(d_out[batch_idx]), op))





