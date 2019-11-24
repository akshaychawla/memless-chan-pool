import torch.nn as nn 
import torch
import random 
from pool_ops import FusedMultiPool
import numpy as np
import time, os, sys
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
from tensorboardX import SummaryWriter

# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# torch.manual_seed(1)
# random.seed(1) 
# np.random.seed(1)


class PRCNv1(nn.Module):
    def __init__(self, nChannels, outchannels, G, exp, kernel_size, padding, stride, randomList=None):
        super(PRCNv1, self).__init__()

        self.exp = exp
        self.G = G
        self.maxpool_size = G
        self.avgpool_size = int((nChannels*exp)/G)
        self.expansion = outchannels*nChannels*self.exp
        self.conv1 = nn.Conv2d(nChannels, outchannels*nChannels, kernel_size=kernel_size, groups=nChannels, padding=padding, bias=True)

        self.transpool1 = nn.MaxPool3d((self.maxpool_size, 1, 1))
        self.transpool2 = nn.AvgPool3d((self.avgpool_size, 1, 1))

        self.index = torch.LongTensor(self.expansion).cuda()
        if isinstance(randomList, np.ndarray):
            self.randomList = randomList.tolist()
        else: 
            self.randomList = list(range(self.expansion))
            random.shuffle(self.randomList)

        for ii in range(self.expansion):
            self.index[ii] = self.randomList[ii]


    def forward(self, x):
        out_conv = self.conv1(x) # nChannels -> outchannels*nChannels
        out = out_conv.repeat(1,self.exp,1,1)  # outchannels*nChannels-> exp*outchannels*nChannels
        out = out[:,self.index,:,:] # randomization
        out = self.transpool1(out) # exp*outchannels*nChannels -> exp*outchannels*nChannels/G
        out = self.transpool2(out) # exp*outchannels*nChannels/(G*meanpool) -> outchannels

        return out, out_conv

class PRCNv2(nn.Module):
    def __init__(self, nChannels, outChannels, G, exp, kernel_size, padding, stride, randomList=None):
        super(PRCNv2, self).__init__()

        # check for valid G,exp ; note: fails for exp,g = (3,4) 
        # assert (float((nChannels*exp))/G - int( (nChannels*exp)/G ) > 0.0), "Incorrect combination of exp and G"

        self.nChannels = nChannels
        self.outChannels = outChannels
        self.G = G 
        self.exp = exp
        self.channel_idx_sets = self.create_channel_idx_sets(randomList)
        self.conv1 = nn.Conv2d(nChannels, outChannels*nChannels, kernel_size=kernel_size, groups=nChannels, padding=padding, bias=True)
        self.fused_pool = FusedMultiPool(self.channel_idx_sets)
        self.avgpool_size = int((nChannels*exp)/G)
        self.transpool2 = nn.AvgPool3d((self.avgpool_size, 1, 1))

    
    def create_channel_idx_sets(self, randomList):

        if not isinstance(randomList, np.ndarray): 
            randomList = list(range(self.nChannels * self.outChannels * self.exp))
            random.shuffle(randomList)
            randomList = np.array(randomList)
        elif isinstance(randomList, np.ndarray):
            randomList = randomList
        else: 
            raise("Type of randomList not understood")
            
        NUM_CHANNEL_SETS = int((self.outChannels * self.nChannels * self.exp) / self.G)
        channel_idx_sets = randomList.reshape((NUM_CHANNEL_SETS,self.G)).astype(np.int32)
        channel_idx_sets = np.mod(channel_idx_sets, self.outChannels * self.nChannels )
        channel_idx_sets = torch.from_numpy(channel_idx_sets).cuda() 
        return channel_idx_sets
    
    def forward(self, x): 
        out = self.conv1(x) 
        out = self.fused_pool(out) # fused pooling
        out = self.transpool2(out)
        return out

class PRCNv1_noconv(nn.Module):
    def __init__(self, nChannels, outchannels, G, exp, kernel_size, padding, stride, randomList=None):
        super(PRCNv1_noconv, self).__init__()

        self.exp = exp
        self.G = G
        self.maxpool_size = G
        self.avgpool_size = int((nChannels*exp)/G)
        self.expansion = outchannels*nChannels*self.exp
        # self.conv1 = nn.Conv2d(nChannels, outchannels*nChannels, kernel_size=kernel_size, groups=nChannels, padding=padding, bias=True)

        self.transpool1 = nn.MaxPool3d((self.maxpool_size, 1, 1))
        self.transpool2 = nn.AvgPool3d((self.avgpool_size, 1, 1))

        self.index = torch.LongTensor(self.expansion).cuda()
        self.randomList = randomList.tolist()

        for ii in range(self.expansion):
            self.index[ii] = self.randomList[ii]

    def forward(self, x_from_conv):
        # out = self.conv1(x) # nChannels -> outchannels*nChannels
        out = x_from_conv.repeat(1,self.exp,1,1)  # outchannels*nChannels-> exp*outchannels*nChannels
        out = out[:,self.index,:,:] # randomization
        out = self.transpool1(out) # exp*outchannels*nChannels -> exp*outchannels*nChannels/G
        out = self.transpool2(out) # exp*outchannels*nChannels/(G*meanpool) -> outchannels
        return out

class PRCNv2_noconv(nn.Module):
    def __init__(self, nChannels, outChannels, G, exp, kernel_size, padding, stride, randomList=None):
        super(PRCNv2_noconv, self).__init__()

        # check for valid G,exp ; note: fails for exp,g = (3,4) 
        # assert (float((nChannels*exp))/G - int( (nChannels*exp)/G ) > 0.0), "Incorrect combination of exp and G"

        self.nChannels = nChannels
        self.outChannels = outChannels
        self.G = G 
        self.exp = exp
        self.channel_idx_sets = self.create_channel_idx_sets(randomList)
        # self.conv1 = nn.Conv2d(nChannels, outChannels*nChannels, kernel_size=kernel_size, groups=nChannels, padding=padding, bias=True)
        self.fused_pool = FusedMultiPool(self.channel_idx_sets)
        self.avgpool_size = int((nChannels*exp)/G)
        self.transpool2 = nn.AvgPool3d((self.avgpool_size, 1, 1))

    
    def create_channel_idx_sets(self, randomList):

        if not isinstance(randomList, np.ndarray): 
            randomList = list(range(self.nChannels * self.outChannels * self.exp))
            random.shuffle(randomList)
            randomList = np.array(randomList)
        elif isinstance(randomList, np.ndarray):
            randomList = randomList
        else: 
            raise("Type of randomList not understood")
            
        NUM_CHANNEL_SETS = int((self.outChannels * self.nChannels * self.exp) / self.G)
        channel_idx_sets = randomList.reshape((NUM_CHANNEL_SETS,self.G)).astype(np.int32)
        channel_idx_sets = np.mod(channel_idx_sets, self.outChannels * self.nChannels )
        channel_idx_sets = torch.from_numpy(channel_idx_sets).cuda() 
        return channel_idx_sets
    
    def forward(self, x_from_conv): 
        # out = self.conv1(x) 
        out = self.fused_pool(x_from_conv) # fused pooling
        out = self.transpool2(out)
        return out

class SimpleNet_v1(nn.Module):
    def __init__(self,):
        super(SimpleNet_v1, self).__init__()

        # self.conv1 = PRCNv1(3, 16, G=4, exp=4, kernel_size=3, padding=1, stride=1)
        self.conv1 = PRCNv2(3, 16, G=4, exp=4, kernel_size=3, padding=1, stride=1)
        # self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
        self.bnorm1 = nn.BatchNorm2d(16)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d((2,2))

        # self.conv2 = PRCNv1(16, 32, G=4, exp=4, kernel_size=3, padding=1, stride=1)
        self.conv2 = PRCNv2(16, 32, G=4, exp=4, kernel_size=3, padding=1, stride=1)
        # self.conv2 = PRCNv2(16, 32, G=4, exp=4, kernel_size=3, padding=1, stride=1)
        # self.conv2 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
        self.bnorm2 = nn.BatchNorm2d(32)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d((2,2))

        # self.conv3 = PRCNv1(32, 32, G=4, exp=4, kernel_size=3, padding=1, stride=1)
        self.conv3 = PRCNv2(32, 32, G=4, exp=4, kernel_size=3, padding=1, stride=1)
        # self.conv3 = PRCNv2(16, 32, G=4, exp=4, kernel_size=3, padding=1, stride=1)
        # self.conv3 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.bnorm3 = nn.BatchNorm2d(32)
        self.relu3 = nn.ReLU()

        # self.conv4 = PRCNv1(32, 32, G=4, exp=4, kernel_size=3, padding=1, stride=1)
        self.conv4 = PRCNv2(32, 32, G=4, exp=4, kernel_size=3, padding=1, stride=1)
        # self.conv4 = PRCNv2(16, 32, G=4, exp=4, kernel_size=3, padding=1, stride=1)
        # self.conv4 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.bnorm4 = nn.BatchNorm2d(32)
        self.relu4 = nn.ReLU()

        # self.conv5 = PRCNv1(32, 32, G=4, exp=4, kernel_size=3, padding=1, stride=1)
        self.conv5 = PRCNv2(32, 32, G=4, exp=4, kernel_size=3, padding=1, stride=1)
        # self.conv5 = PRCNv2(32, 32, G=4, exp=4, kernel_size=3, padding=1, stride=1)
        # self.conv5 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.bnorm5 = nn.BatchNorm2d(32)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d((2,2))

        self.fc1 = nn.Linear(512, 256)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)


    def forward(self, x):

        out = self.conv1(x) 
        out = self.bnorm1(out)
        out = self.relu1(out) 
        out = self.pool1(out)

        out = self.conv2(out) 
        out = self.bnorm2(out)
        out = self.relu2(out) 
        out = self.pool2(out)

        out = self.conv3(out) 
        out = self.bnorm3(out)
        out = self.relu3(out) 

        out = self.conv4(out) 
        out = self.bnorm4(out)
        out = self.relu4(out) 

        out = self.conv5(out) 
        out = self.bnorm5(out)
        out = self.relu5(out) 
        out = self.pool5(out)

        out = torch.flatten(out, start_dim=1)

        out = self.fc1(out) 
        out = self.relu4(out)
        out = self.fc2(out)

        return out 


def train(): 

    model = SimpleNet_v1().cuda() 
    transform = transforms.Compose( [transforms.ToTensor()])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=16,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=16,
                                            shuffle=False, num_workers=2)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    def sc_fun(ep):
        if ep>=225:
            return 0.001 
        elif ep>=150:
            return 0.01 
        else: 
            return 0.1
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, sc_fun)
    epochs = 300 
    writer = SummaryWriter()

    for epoch in range(epochs):  # loop over the dataset multiple times

        # train loop
        model.train()
        scheduler.step()
        running_loss = 0.0
        num_samples_iterated = 0
        for i, data in enumerate(trainloader, 0):
            # get the inputs
            inputs, labels = data
            inputs = inputs.cuda() 
            labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            num_samples_iterated += outputs.shape[0]
        
        # testing loop 
        model.eval() 
        true_y = [] 
        pred_y = []
        num_samples_iterated = 0
        for i, data in enumerate(testloader, 0):
            inputs, labels = data 
            inputs = inputs.cuda() 
            labels = labels.cuda() 
            outputs = model(inputs)
            
            pred_labels = torch.argmax(outputs, 1)
            true_y.extend(list(labels.detach().cpu().numpy()))
            pred_y.extend(list(pred_labels.detach().cpu().numpy()))
            num_samples_iterated += inputs.shape[0]
        
        true_y = np.array(true_y)
        pred_y = np.array(pred_y)

        accuracy = np.count_nonzero(true_y==pred_y)
        accuracy = accuracy / num_samples_iterated 
        
        print("Epoch {} | Average loss: {} | Test acc: {}".format(epoch+1, running_loss/num_samples_iterated, accuracy))
        writer.add_scalar("loss", running_loss/num_samples_iterated, epoch+1)
        writer.add_scalar("acc", accuracy, epoch+1)

        

if __name__ == "__main__":
    train()

