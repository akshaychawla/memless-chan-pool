import argparse 

def cli(): 
    parser = argparse.ArgumentParser("PRC-NPTN networks") 
    parser.add_argument("--init_lr", type=float,  default=0.1) 
    parser.add_argument("--lr_schedule", type=int, nargs="+", default=[150, 225])
    parser.add_argument("--momentum", type=float,  default=0.9)
    parser.add_argument("--epochs", type=int,  default=100)
    parser.add_argument("--batch_size", type=int,  default=32)
    parser.add_argument("--display_iter", type=int,  default=50)
    parser.add_argument("--save_iter", type=int,  default=10)
    parser.add_argument("--model", type=str,  default="densenet", choices=["densenet", "denseprc"]) 
    parser.add_argument("--weight_decay", type=float,  default=1e-05) 
    parser.add_argument("--dataset", type=str,  default="cifar", choices=["cifar", "imagenet"])
    parser.add_argument("--CMP", type=int,  default=3)
    parser.add_argument("--G", type=int,  default=12)
    parser.add_argument("--disable-cuda",  action="store_true")
    parser.add_argument("--logs", type=str,  default="./runs/exp-1")
    args = parser.parse_args() 
    return args 

