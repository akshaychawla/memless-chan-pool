import numpy as np 
import matplotlib.pyplot as plt 
import torch.nn as nn 
import torch 
import random 
from PRCN import PRCNv1_noconv, PRCNv2_noconv, PRCNv1, PRCNv2, PRCNPTN, FastPRCNPTN
from copy import deepcopy 
import cupy as cp

# Hyperparameters
ip_chans, op_chans = 8,16 
h,w = 128, 128 
batch_size = 16 
G = 12 
# exp = 3
CMP = 4
print("Performing forward and backward pass checks:") 
print("Parameters: ip_chans: {} | op_chans: {} | batch_size: {} | G: {} | CMP: {}".format(
        ip_chans, op_chans, batch_size, G, CMP))

# Pre-requisites for shape, randomlist
ltest_ip = torch.randn(batch_size, ip_chans, h, w).float().cuda()
ltest = PRCNPTN(ip_chans,op_chans,G=G,CMP=CMP,kernel_size=3,padding=1,stride=1).cuda()
ltest.eval()
ltest_op, ltest_convop = ltest.forward(ltest_ip)
print("PRCNPTN op shape: {} , PRCNPTN Conv layer op shape: {}".format(ltest_op.shape, ltest_convop.shape))

# data from conv layer 
ltest_ip = ltest_ip.detach()
ltest_ip.requires_grad = True
ltest_ip.retain_grad() 
randomList = deepcopy(ltest.randomList)

# # GT to regress against 
gt = torch.rand_like(ltest_op.detach().cpu()).cuda()
mse = nn.MSELoss()

# # no-conv Layers 
layer1 = PRCNPTN(ip_chans, op_chans, G=G, CMP=CMP, kernel_size=3, padding=1, stride=1, randomList=randomList).cuda()
layer2 = FastPRCNPTN(ip_chans, op_chans, G=G, CMP=CMP, kernel_size=3, padding=1, stride=1, randomList=randomList).cuda()
# Copy weights 
layer2.conv1.load_state_dict(layer1.conv1.state_dict())

# Test backward pass PRCNPTN 
op1,_ = layer1(ltest_ip) 
op1.retain_grad()
loss = mse(op1, gt) 
loss.backward()
grad1 = ltest_ip.grad.cpu().detach()
op1_grad = op1.grad.cpu().detach().numpy()
op1_val   = op1.cpu().detach().numpy()
del loss
ltest_ip.grad.zero_()

# # Test backward pass FastPRCNPTN 
op2,_ = layer2(ltest_ip) 
op2.retain_grad()
loss = mse(op2, gt) 
loss.backward()
grad2 = ltest_ip.grad.cpu().detach()
op2_grad = op2.grad.cpu().detach().numpy()
op2_val   = op2.cpu().detach().numpy()
del loss 

print("Forward pass activation equaliy:") 
print(torch.allclose(op2,op1))
print("Backward pass gradient equality:")
print(torch.allclose(grad1, grad2))




