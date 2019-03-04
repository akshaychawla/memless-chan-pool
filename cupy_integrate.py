import numpy as np 
import cupy as cp 
import cupy
import math

def verify_output(d_out, d_in):
    d_out_cpu = cp.asnumpy(d_out)
    d_in_cpu = cp.asnumpy(d_in)
    chanpool = np.amax(d_in_cpu, axis=0)
    print("Same? {}".format(np.allclose(chanpool, d_out_cpu)))

with open("./kernel_test.cu", "rt") as f: 
    code = f.read() 


DEPTH = 128 
HEIGHT = 20 
WIDTH = 24 

max_kernel = cp.RawKernel(code, "KERNEL_max")

d_in = cp.random.randn(DEPTH, HEIGHT,WIDTH, dtype=cp.float32) 
d_in_DIMS = cp.array([DEPTH, HEIGHT, WIDTH], dtype=cp.int32)
d_out_DIMS = cp.array([HEIGHT, WIDTH], dtype=cp.int32)
d_out = cp.zeros(shape=d_out_DIMS.tolist(), dtype=cp.float32)

MAX_TILE_DIM = 16 
NUM_TILES_X = math.ceil(float(WIDTH)/MAX_TILE_DIM)
NUM_TILES_Y = math.ceil(float(HEIGHT)/MAX_TILE_DIM)


grid = (NUM_TILES_X, NUM_TILES_Y)
block = (MAX_TILE_DIM,MAX_TILE_DIM)

# run the kernel 
max_kernel(grid, block, (d_out, d_out_DIMS, d_in, d_in_DIMS))

# print(d_in)
# print(d_out)

# check the final result 
verify_output(d_out, d_in)


# add_kernel = cp.RawKernel(code, "add_2d")

# x1 = cp.random.randn(2,3,4, dtype=cp.float32)
# dimensions = cp.array([2,3,4], dtype=cp.int32)
# print(x1[0])
# x1 = cp.array([
#     [1,2,3],
#     [4,5,6]
# ]).astype(cupy.float32)


# x2 = cp.array([
#     [7,8,9],
#     [10,11,12]
# ]).astype(cupy.float32)

# y = cp.zeros((3,4), dtype=cupy.float32)

# grid = (1,)
# block = (3,4)


# add_kernel(grid, block, (x1,dimensions,y))
# print(y)

