import numpy as np 
import matplotlib.pyplot as plt 
import torch.nn as nn 
import torch
import random 
from pool_ops import FusedMultiPool
import time, os, sys
from PRCN import PRCNv2, PRCNv1 

def main(): 
    layer1 = PRCNv1(16, 32, G=4, exp=3, kernel_size=3, padding=1, stride=1).cuda()
    randomList = layer1.randomList
    layer2 = PRCNv2(16, 32, G=4, exp=3, kernel_size=3, padding=1, stride=1, randomList=np.array(randomList)).cuda()
    x = torch.randn(10, 16, 200, 200).cuda()

    # Copy conv2d parameters from layer1 --> Layer2 
    layer2.conv1.weight = layer1.conv1.weight 
    layer2.conv1.bias = layer1.conv1.bias

    y1 = layer1(x) 
    y2 = layer2(x)
    print(torch.allclose(y1, y2))

    del y1, y2, x

    x = torch.randn(10, 16, 200, 200).cuda()
    y1 = layer1(x) 
    y2 = layer2(x)


    print(torch.allclose(y1, y2))
    import pdb; pdb.set_trace()
    print("..end")



if __name__ == "__main__":
    main()
