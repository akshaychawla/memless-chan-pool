"""
Timing experiments for 2 layer PRC network
"""

import numpy as np 
import torch 
from torch import nn 
from PRCN import PRCNPTN, FastPRCNPTN
BLOCK = PRCNPTN
import copy, os, sys 
import argparse
import time 

# Reproducibility 
torch.manual_seed(0) 
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(0)

class Triplet(nn.Module): 
    def __init__(self, in_channels, out_channels, G, CMP): 
        super(Triplet, self).__init__()
        self.prcn   = BLOCK(in_channels, out_channels, G=G, CMP=CMP, kernel_size=5, stride=1, padding=2, bias=False)
        self.bnorm  = nn.BatchNorm2d(out_channels)
        self.act    = nn.PReLU() 
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self,x): 
        #print("called once")
        out = self.prcn(x) 
        # if out.shape[1] != self.out_channels:
        #     print("Uh oh") 
        #     import pdb; pdb.set_trace()
        # Autocrop to right dims
        # print("Autocrop diff: {}".format(out.shape[1] - self.out_channels)) 
        out = out[:,:self.out_channels]
        out = self.bnorm(out) 
        out = self.act(out)
       
        #out = self.act(self.bnorm(self.prcn(x)))
        return out

class Lenet(nn.Module): 
    def __init__(self, G, CMP, conv_channels, num_hidden, num_triplets=1): 
        super(Lenet, self).__init__() 
        
        # params
        self.num_hidden = num_hidden
        self.G = G 
        self.CMP = CMP 
        self.conv_channels = conv_channels

        # First conv
        self.conv1  = nn.Conv2d(1, conv_channels, kernel_size=5, stride=1, padding=2, bias=True)
        self.bnorm1 = nn.BatchNorm2d(conv_channels)
        self.act1   = nn.PReLU()
        
        # triplets
        in_channels = conv_channels
        out_channels = self.num_hidden
        self.triplets = [] 
        for trip in range(num_triplets):
            #print("triplet params: in:{} out:{}".format(in_channels, out_channels))
            self.triplets.append(Triplet(in_channels, out_channels, G=G, CMP=CMP))
            in_channels = copy.deepcopy(out_channels) 
        self.triplets = nn.Sequential(*self.triplets)
        
        self.pool1  = nn.MaxPool2d(3)
        self.pool2  = nn.MaxPool2d(3)
        
        # Ending
        self.prcn3  = BLOCK(self.num_hidden, 20, G=G, CMP=CMP, kernel_size=3, stride=1, padding=1, bias=False)
        self.bnorm3 = nn.BatchNorm2d(20)
        self.act3   = nn.PReLU()
        
        self.linear = nn.Linear(20*3*3, 10) 

    def forward(self, x): 
        out = self.act1(self.bnorm1(self.conv1(x)))
        out = self.pool1(out)
        out = self.triplets(out)
        out = self.act3(self.bnorm3(self.prcn3(out)))
        out = self.pool2(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out

def cli(): 
    parser = argparse.ArgumentParser() 
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--G", type=int, default=12)
    parser.add_argument("--CMP", type=int, default=4)
    parser.add_argument("--num_hidden", type=int, default=36)
    parser.add_argument("--num_triplets", type=int, default=1)
    parser.add_argument("--conv_channels", type=int, default=48)
    parser.add_argument("--datapoints", type=int, default=50)
    parser.add_argument("--outfile", type=str, default="timingresults.txt")
    args = parser.parse_args()
    return args

def test(): 
    # import pdb; pdb.set_trace()
    # net = Lenet(G=12, CMP=4, num_hidden=36, num_triplets=2).cuda()
    # _x = torch.randn(2, 1, 28, 28).cuda()
    # y = net(_x) 
    # print(y.shape)

    global BLOCK

    print("Timing experiment for slow")

    args = cli()
    print(args)
    
    net = Lenet(
        G=args.G, 
        CMP=args.CMP, 
        num_hidden=args.num_hidden, 
        num_triplets=args.num_triplets,
        conv_channels=args.conv_channels
    )
    net = net.cuda()
    net.eval()
   
    # Memory code 
    torch.cuda.empty_cache()
    start_mem = torch.cuda.memory_allocated() 
    #assert float(start_mem) < 0.000001, "Start mem: {}".format(start_mem)
    _x = torch.randn(args.batchsize, 1, 28, 28).cuda()
    with torch.no_grad():
        y = net(_x) 
    torch.cuda.synchronize()
    end_mem   = torch.cuda.memory_allocated()
    slow_mem = (end_mem - start_mem) / (1024.0*1024.0)
    torch.cuda.empty_cache()
    del y, _x  
    torch.cuda.empty_cache()



    # timing code 
    _x = torch.randn(args.batchsize, 1, 28, 28).cuda()
    slow_elapsed = [] 
    slow_memory  = []
    for idx in range(args.datapoints): 
        # print(idx)
        #start_event = torch.cuda.Event(enable_timing=True)
        #end_event   = torch.cuda.Event(enable_timing=True)
        
        start = time.time()

        #start_event.record()
        
        with torch.no_grad():
            y = net(_x) 
        
        # end_event.record()
        torch.cuda.synchronize()
        slow_elapsed.append(time.time() - start)
        del y
        #elapsed.append(start_event.elapsed_time(end_event))
        
        #mem = torch.cuda.memory_stats()["allocated_bytes.all.peak"] / (1024*1024) # bytes --> kilobytes --> Megabytes (pokemon?)
        #mem = torch.cuda.max_memory_allocated(0)
        #mem = mem / (1024.0*1024.0)
        #memory.append(mem)
       
    if len(slow_elapsed)>1:
        slow_elapsed = slow_elapsed[5:]
    slow_elapsed = np.mean(slow_elapsed)
    del _x
    del net
    torch.cuda.empty_cache()
    
    #print("Avg: {} Std: {}".format(np.mean(elapsed), np.std(elapsed)))
    #print("max Memory required: {:.2f}".format(max(memory))) 

    #############################################################################

    print("Timing experiment for fast")

    BLOCK = FastPRCNPTN
     
    net = Lenet(
        G=args.G, 
        CMP=args.CMP, 
        num_hidden=args.num_hidden, 
        num_triplets=args.num_triplets,
        conv_channels=args.conv_channels
    )
    net = net.cuda()
    net.eval()

    # Memory code 
    start_mem = torch.cuda.memory_allocated() 
    #assert float(start_mem) < 0.000001, "Start mem: {}".format(start_mem)
    _x = torch.randn(args.batchsize, 1, 28, 28).cuda()
    with torch.no_grad():
        y = net(_x) 
    torch.cuda.synchronize()
    end_mem   = torch.cuda.memory_allocated()
    fast_mem = (end_mem - start_mem) / (1024.0*1024.0)
    torch.cuda.empty_cache()
    del y, _x  
    torch.cuda.empty_cache()

    # timing code 
    _x = torch.randn(args.batchsize, 1, 28, 28).cuda()
    fast_elapsed = [] 
    fast_memory  = []
    for idx in range(args.datapoints): 
        # print(idx)
        #start_event = torch.cuda.Event(enable_timing=True)
        #end_event   = torch.cuda.Event(enable_timing=True)
        
        start = time.time()

        #start_event.record()
        
        with torch.no_grad():
            y = net(_x) 
        
        # end_event.record()
        torch.cuda.synchronize()
        fast_elapsed.append(time.time() - start)
        del y
        #elapsed.append(start_event.elapsed_time(end_event))
        
        #mem = torch.cuda.memory_stats()["allocated_bytes.all.peak"] / (1024*1024) # bytes --> kilobytes --> Megabytes (pokemon?)
        #mem = torch.cuda.max_memory_allocated(0)
        #mem = mem / (1024.0*1024.0)
        #memory.append(mem)
    
    del _x, net
    if len(fast_elapsed)>1:
        fast_elapsed = fast_elapsed[5:]
    fast_elapsed = np.mean(fast_elapsed)
    #print("Avg: {} Std: {}".format(np.mean(fast_elapsed), np.std(fast_elapsed)))
    #print("max Memory required: {:.2f}".format(max(memory))) 

    print("Speedup: {}".format(slow_elapsed / fast_elapsed))
    print("Memory efficiency: {}".format(slow_mem / fast_mem))


    with open(args.outfile, "at") as f: 
        f.write(str(args)+"\n")
        f.write("speedup: {}".format(slow_elapsed / fast_elapsed))
        f.write("memory efficiency: {}".format(slow_mem/fast_mem))
        #f.write("Avg: {} Std: {}\n".format(np.mean(elapsed), np.std(elapsed)))
        #f.write("max memory required: {}".format(max(memory)))
        f.write("\n\n")

if __name__ == "__main__": 
    test()
