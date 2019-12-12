import numpy as np
import torch.nn as nn
import torch
import random
from PRCN import PRCNv1, PRCNv2
import os, sys, time
from pprint import pprint
import argparse

def runTest(params):
    """
    Run performance test with given parameters for PRCNv1 and PRCNv2
    """
    ip_chans, op_chans = params["ip_chans"], params["op_chans"]
    h, w = params["h"], params["w"]
    batch_size = params["batch_size"]
    G, exp = params["G"], params["exp"]
    module = None
    if params["module"] == "PRCNv1": 
        module = PRCNv1
    elif params["module"] == "PRCNv2": 
        module = PRCNv2 
    else: 
        raise NotImplementedError("module {} cannot be tested".format(params["module"]))

    obj = module(
                ip_chans, op_chans,
                G=G, exp=exp, kernel_size=3, 
                padding=1, stride=1
            ) 
    obj = obj.cuda()
    obj.eval() 

    x = torch.randn(batch_size, ip_chans, h, w).float().cuda() 
    # Pre-cache run 
    for _ in range(5):
        with torch.no_grad():
            _ = obj(x) 
    # Timed run 
    elapsed = [] 
    for _ in range(params["numsamples"]): 
        with torch.no_grad(): 
            start = time.time()  
            _ = obj(x) 
            elapsed.append(time.time() - start)
    return float(np.mean(elapsed))

def cli(): 
    parser = argparse.ArgumentParser() 
    params = {
        "module": "PRCNv2",
        "ip_chans":16, 
        "op_chans":64,
        "h": 128, "w": 128, 
        "batch_size":16, 
        "G": 4, "exp":3,
        "numsamples": 50
        } 
    for key,val in params.items():
        parser.add_argument("--"+key, type=type(val), default=val)
    args = parser.parse_args() 
    return vars(args)

if __name__ == "__main__":
    args = cli() 
    pprint(args)
    elapsed = runTest(args)
    print("mean time: {:.8f}".format(elapsed))
    mem = torch.cuda.max_memory_allocated(0)
    mem = mem / (1024.0*1024.0)
    print("max mem: {:.2f}".format(mem))



