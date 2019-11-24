import numpy as np 
import matplotlib.pyplot as plt 
import torch.nn as nn 
import torch 
import random 
from PRCN import PRCNv1_noconv, PRCNv2_noconv, PRCNv1, PRCNv2 
from copy import deepcopy 
import cupy as cp

# Hyperparameters
ip_chans, op_chans = 8,16 
h,w = 128, 128 
batch_size = 16 
G = 4 
exp = 3
print("Performing forward and backward pass checks:") 
print("Parameters: ip_chans: {} | op_chans: {} | batch_size: {} | G: {} | exp: {}".format(
        ip_chans, op_chans, batch_size, G, exp))

# Pre-requisites for shape, randomlist
ltest_ip = torch.randn(batch_size, ip_chans, h, w).float().cuda()
ltest = PRCNv1(ip_chans,op_chans,G=G,exp=exp,kernel_size=3,padding=1,stride=1).cuda()
ltest.eval()
ltest_op, ltest_convop = ltest.forward(ltest_ip)
print("PRCN op shape: {} , PRCN Conv layer op shape: {}".format(ltest_op.shape, ltest_convop.shape))

# data from conv layer 
x_conv = ltest_convop.detach()
x_conv.requires_grad = True
x_conv.retain_grad() 
randomList = np.array(ltest.randomList)

# GT to regress against 
gt = torch.rand_like(ltest_op.detach().cpu()).cuda()
mse = nn.MSELoss()

# no-conv Layers 
layer1 = PRCNv1_noconv(ip_chans, op_chans, G=G, exp=exp, kernel_size=3, padding=1, stride=1, randomList=randomList).cuda() 
layer2 = PRCNv2_noconv(ip_chans, op_chans, G=G, exp=exp, kernel_size=3, padding=1, stride=1, randomList=randomList).cuda() 

# Test backward pass prcn v1 
op1 = layer1(x_conv) 
op1.retain_grad()
loss = mse(op1, gt) 
loss.backward()
grad1 = x_conv.grad.cpu().detach()
op1_grad = op1.grad.cpu().detach().numpy()
op1_val   = op1.cpu().detach().numpy()
del loss
x_conv.grad.zero_()

# Test backward pass prcn v2
op2 = layer2(x_conv) 
op2.retain_grad()
loss = mse(op2, gt) 
loss.backward()
grad2 = x_conv.grad.cpu().detach()
op2_grad = op2.grad.cpu().detach().numpy()
op2_val   = op2.cpu().detach().numpy()
del loss 

print("Forward pass activation equaliy:") 
print(torch.allclose(op2,op1))
print("Backward pass gradient equality:")
print(torch.allclose(grad1, grad2))




