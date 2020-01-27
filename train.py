import numpy as np 
import matplotlib.pyplot as plt 
from utils import cli 
import torch 
from torch import nn
import torchvision
from torchvision import transforms
from densenet import densenet_cifar
from denseprc import denseprc_cifar_dipan
import copy, sys, time, os
from tensorboardX import SummaryWriter

def train(net, optimizer, scheduler, lossfun, dataloader, writer, args): 
    """
    Training loop
    """
    net.train()
    for epoch in range(args.epochs): 
        print("Epoch: {}".format(epoch+1))
        losses = []
        accuracies = []
        for batch_idx, (inputs, targets) in enumerate(dataloader): 

            # forward pass
            inputs, targets = inputs.to(args.device), targets.to(args.device)
            outputs = net(inputs) 
            loss    = lossfun(outputs, targets) 

            # backprop + grad update
            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step()
            
            outputs = torch.argmax(outputs, dim=1)
            accuracy = float( torch.sum(outputs==targets).item() / len(targets))
            accuracies.append(accuracy) 
            losses.append(loss.item()) 

            if (batch_idx % args.display_iter) == 0: 
                print("iter: {} loss: {} err: {}".format(batch_idx+1, losses[-1], 1.0-accuracies[-1]))
            
            if (batch_idx % args.save_iter) == 0: 
                writer.add_scalar("train/iterLoss", losses[-1], epoch*len(dataloader)+batch_idx)
                writer.add_scalar("train/iterError", 1.0-accuracies[-1], epoch*len(dataloader)+batch_idx)
        
        print("avg loss: {} err: {}".format(np.mean(losses), 1.0-np.mean(accuracies)))
        writer.add_scalar("train/epochLoss", np.mean(losses), epoch)
        writer.add_scalar("train/epochError", 1.0-np.mean(accuracies), epoch)
        scheduler.step()

def test(net, lossfun, dataloader, writer, args): 
    """
    Test 
    """
    net.eval()
    losses = [] 
    accuracies = [] 
    for batch_idx, (inputs, targets) in enumerate(dataloader): 
        inputs, targets = inputs.to(args.device), targets.to(args.device) 
        outputs = net(inputs) 
        loss    = lossfun(outputs, targets) 
        outputs = torch.argmax(outputs, dim=1) 
        accuracy = float( torch.sum(outputs==targets).item() / len(targets))
        accuracies.append(accuracy) 
        losses.append(loss.item()) 
    print("[TEST] loss: {} error: {}".format(np.mean(losses), 1.0-np.mean(accuracies)))
    writer.add_scalar("test/loss", np.mean(losses), int(args.epochs)) 
    writer.add_scalar("test/error", 1.0-np.mean(accuracies), int(args.epochs))

def run(): 

    # args
    args = cli() 
    print("run with args: {}".format(args))

    # logs 
    writer = SummaryWriter(args.logs)

    # Store args 
    with open(os.path.join(args.logs, "args.txt"), "wt") as f: 
        f.write(str(args)+"\n")

    # model
    args.device = None
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
    else:
        args.device = torch.device('cpu')
    if args.model == "densenet": 
        net = densenet_cifar() 
        net = net.to(device=args.device)
    elif args.model == "denseprc": 
        net = denseprc_cifar_dipan(G=args.G, CMP=args.CMP, reduction=1.0) 
        net = net.to(device=args.device)
    else:
        raise NotImplementedError("{} is not available".format(args.model))

    # data
    trainset = torchvision.datasets.CIFAR10(
        root="./data", 
        train=True, 
        download=True, 
        transform=transforms.Compose
            (
            [transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
            )
        )
    testset = torchvision.datasets.CIFAR10(
        root="./data",
        train=False, 
        download=True, 
        transform=transforms.Compose
            (
            [transforms.ToTensor(), 
             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
            )
        )
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    testloader  = torch.utils.data.DataLoader(testset,  batch_size=args.batch_size, shuffle=True, num_workers=2)
    classes = copy.deepcopy(trainset.classes)
    
    # optimizer + loss
    optimizer = torch.optim.SGD(
                params=net.parameters(),  
                lr=args.init_lr, 
                momentum=args.momentum, 
                weight_decay=args.weight_decay)
    xent  = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=args.lr_schedule,
                gamma=0.1
                )

    # train+test
    train(net, optimizer, scheduler, xent, trainloader, writer, args)
    test(net, xent, testloader, writer, args)
    writer.close()

if __name__ == "__main__": 
    run()





