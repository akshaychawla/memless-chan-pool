import torch.nn as nn 
import torch
import random 
from pool_ops import FusedMultiPool
import numpy as np
import time, os, sys

class RPTNv1(nn.Module):

    def __init__(self,  in_ch, out_ch, G, k, pad, stride ):
        super(RPTNv1, self ).__init__()

        self.conv1 = nn.Conv2d(in_ch, G*in_ch*out_ch, kernel_size=k, padding=pad, groups=in_ch, stride=stride, bias=True)
        self.transpool = nn.MaxPool3d((G, 1, 1))
        self.expansion = 4
        self.conv2 = nn.Conv2d(in_ch * out_ch*self.expansion, out_ch, kernel_size=1, stride=1, groups=1)
        self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, groups=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.m2 = nn.PReLU()
        self.m3 = nn.PReLU()


        self.out_ch = out_ch
        self.G = G
        self.index = torch.LongTensor(in_ch*out_ch*G*self.expansion).cuda()
        self.randomlist = list(range(in_ch*out_ch*G*self.expansion))
        random.shuffle(self.randomlist)

        for ii in range(in_ch*out_ch*G*self.expansion):
            self.index[ii] = self.randomlist[ii]

    def forward(self, x):

        out1 = self.conv1(x) # in_ch -> G*in_ch*out_ch
        start = time.time()
        out = out1.repeat(1,self.expansion,1,1)  # G*in_ch*out_ch -> G*in_ch*out_ch*expansion
        out = out[:,self.index,:,:] # randomization
        out = self.transpool(out)  # G*in_ch*out_ch*expansion  ->  in_ch*out_ch*expansion
        end = time.time()
        print("v1 pool took {}".format(end - start))
        # out = self.conv2(out) # in_ch*out_ch*expansion  ->  out_ch
        # out = self.m2(out)
        # out = self.conv3(out) # out_ch  ->  out_ch

        return out

class RPTNv2(nn.Module):

    def __init__(self,  in_ch, out_ch, G, k, pad, stride, channel_idx_sets):
        super(RPTNv2, self ).__init__()

        self.conv1 = nn.Conv2d(in_ch, G*in_ch*out_ch, kernel_size=k, padding=pad, groups=in_ch, stride=stride, bias=True)
        self.fused_pool = FusedMultiPool(channel_idx_sets)

    def forward(self, x):

        out = self.conv1(x) # in_ch -> G*in_ch*out_ch
        start = time.time()
        out = self.fused_pool(out) # fused pooling
        end = time.time()
        print("v2 pool took {}".format(end - start))

        return out 


def test():

    rpn_layer_v1 = RPTNv1(in_ch=16, out_ch=32, G=4, k=3, pad=1, stride=1).cuda()
    randomList = np.array(rpn_layer_v1.randomlist)
    expansion = rpn_layer_v1.expansion
    NUM_CHANNEL_SETS = int(len(randomList) / expansion)
    channel_idx_sets = randomList.reshape((NUM_CHANNEL_SETS,expansion)).astype(np.int32)
    channel_idx_sets = np.mod(channel_idx_sets, NUM_CHANNEL_SETS)
    channel_idx_sets = torch.from_numpy(channel_idx_sets).cuda() 
    rpn_layer_v2 = RPTNv2(in_ch=16, out_ch=32, G=4, k=3, pad=1, stride=1, channel_idx_sets=channel_idx_sets).cuda()

    ip = torch.randn((2,16,128,128)).cuda()

    time_v1, time_v2 = [], []
    for i in range(20):
        start = time.time()
        y_rptnv1 = rpn_layer_v1.forward(ip)
        time_v1.append(time.time()-start)

    print("\n\n\n")

    for i in range(20):
        start = time.time()
        y_rptnv2 = rpn_layer_v2.forward(ip)
        time_v2.append(time.time()-start)
        # print("Full thing took ", time_v2[-1])
    
    time_v1 = np.array(time_v1)
    time_v2 = np.array(time_v2) 
    print(np.mean(time_v1[5:]))
    print(np.mean(time_v2[5:]))





if __name__ == "__main__":
    test()