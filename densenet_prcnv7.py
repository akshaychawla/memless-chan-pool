import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torchvision.models as models

import sys
import math
import random

# current best
class PRCN_v5_Bottleneck_currentbest(nn.Module):
	def __init__(self, in_planes, growth_rate):
		super(PRCN_v5_Bottleneck_currentbest, self).__init__()

		# growth rate needs to be even
		# growth_rate /=2
		self.in_planes = in_planes
		self.growth_rate = growth_rate
		interChannels = 4*growth_rate
		self.bn1 = nn.BatchNorm2d(in_planes)
		self.conv1 = nn.Conv2d(in_planes, interChannels, kernel_size=1,
							   bias=False, groups=1)
		self.bn2 = nn.BatchNorm2d(interChannels)
		self.conv2 = nn.Conv2d(interChannels, growth_rate, kernel_size=3,
							   padding=1, bias=False)

		# self.conv_pre = nn.Conv2d(in_planes, growth_rate, kernel_size=1, bias=False)
		G = 4
		# self.bn1 = nn.BatchNorm2d(in_planes)
		self.bn3 = nn.BatchNorm2d(growth_rate)
		self.conv1_prcn = nn.Conv2d(growth_rate, growth_rate, kernel_size=3, padding=1, groups=growth_rate, bias=False)
		# self.bn2 = nn.BatchNorm2d(4*growth_rate)
		# self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

		self.transpool = nn.MaxPool3d((G, 1, 1))
		self.index = torch.LongTensor(growth_rate*G).cuda()
		self.randomlist = range(growth_rate*G)
		random.shuffle(self.randomlist)

		for ii in range(growth_rate*G):
			self.index[ii] = self.randomlist[ii]

		# self.channelpassindex = torch.LongTensor(in_planes - 2*growth_rate).cuda()
		# self.randomlist = range(growth_rate*G)

		FC_k = 1
		FC_pad = 0

		# self.conv_fc_1 = nn.Conv2d(growth_rate, growth_rate, kernel_size=FC_k, padding=FC_pad, groups=1, stride=1, bias=False)
		# self.conv_fc_2 = nn.Conv2d(growth_rate, growth_rate, kernel_size=FC_k, padding=FC_pad, groups=1, stride=1, bias=False)
		

		# self.m_fc_1 = nn.ReLU()
		# self.m_fc_2 = nn.ReLU()
		# # self.m = nn.ReLU()
		# self.bn_fc_1 = nn.BatchNorm2d(growth_rate)
		# self.bn_fc_2 = nn.BatchNorm2d(growth_rate)


	def forward(self, x):
		out = self.conv1(F.relu(self.bn1(x))) # in_ch -> 4GR
		out_dc = self.conv2(F.relu(self.bn2(out))) # 4GR -> GR
		out = self.conv1_prcn(F.relu(self.bn3(out_dc))) # GR -> G*GR
		# out = self.conv1(out) # in_ch -> G*in_ch
		out = out[:,self.index,:,:] # randomization
		out = self.transpool(out)  # G*GR  ->  GR

		# out = self.bn_fc_1(out)
		# out = self.m_fc_1(out)
		# out = self.conv_fc_1(out) # in_ch -> fc_ch
		# # print(out.cpu().size())

		# out = self.bn_fc_2(out)
		# out = self.m_fc_2(out)
		# out = self.conv_fc_2(out) # fc_ch -> fc_ch
		# print(out.cpu().size())
		# print(self.in_planes)
		# print(self.growth_rate)
		out = torch.cat((out, out_dc, x[:,:self.in_planes - self.growth_rate ,:,:]),1) # in_ch -> in_ch + GR      
		# print(out.cpu().size())
		# [out_prcn, out_dc, skip]
		return out






########################################################################################
########################################################################################
########################################################################################

class Bottleneck(nn.Module):
	def __init__(self, nChannels, growthRate):
		super(Bottleneck, self).__init__()
		interChannels = 4*growthRate
		self.bn1 = nn.BatchNorm2d(nChannels)
		self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
							   bias=False)
		self.bn2 = nn.BatchNorm2d(interChannels)
		self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
							   padding=1, bias=False)

		# self.drop1 = nn.Dropout2d(p=0.2)
		# self.drop2 = nn.Dropout2d(p=0.2)

	def forward(self, x):
		out = self.conv1(F.relu(self.bn1(x)))
		# out = self.drop1(out)

		out = self.conv2(F.relu(self.bn2(out)))
		# out = self.drop2(out)
		
		out = torch.cat((x, out), 1)
		return out

class SingleLayer(nn.Module):
	def __init__(self, nChannels, growthRate):
		super(SingleLayer, self).__init__()
		self.bn1 = nn.BatchNorm2d(nChannels)
		self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
							   padding=1, bias=False)

	def forward(self, x):
		out = self.conv1(F.relu(self.bn1(x)))
		out = torch.cat((x, out), 1)
		return out

class Transition(nn.Module):
	def __init__(self, nChannels, nOutChannels):
		super(Transition, self).__init__()
		self.bn1 = nn.BatchNorm2d(nChannels)
		self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
							   bias=False)
		# self.drop1 = nn.Dropout2d(p=0.2)

	def forward(self, x):
		out = self.conv1(F.relu(self.bn1(x)))
		# out = self.drop1(out)
		
		out = F.avg_pool2d(out, 2)
		return out



########################################################################################
########################################################################################
########################################################################################

class PRCN_v5_Bottleneck(nn.Module):
	def __init__(self, in_planes, growth_rate):
		super(PRCN_v5_Bottleneck, self).__init__()

		# growth rate needs to be even
		# growth_rate /=2
		self.in_planes = in_planes
		self.growth_rate = growth_rate
		interChannels = 4*growth_rate
		self.bn1 = nn.BatchNorm2d(in_planes)
		self.conv1 = nn.Conv2d(in_planes, interChannels, kernel_size=1,
							   bias=False, groups=1)
		self.bn2 = nn.BatchNorm2d(interChannels)
		self.conv2 = nn.Conv2d(interChannels, growth_rate, kernel_size=3,
							   padding=1, bias=False)

		# self.conv_pre = nn.Conv2d(in_planes, growth_rate, kernel_size=1, bias=False)
		G = 4
		# self.bn1 = nn.BatchNorm2d(in_planes)
		self.bn3 = nn.BatchNorm2d(growth_rate)
		self.conv1_prcn = nn.Conv2d(growth_rate, G*growth_rate, kernel_size=3, padding=1, groups=growth_rate, bias=False)
		# self.bn2 = nn.BatchNorm2d(4*growth_rate)
		# self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

		self.transpool = nn.MaxPool3d((G, 1, 1))
		self.index = torch.LongTensor(growth_rate*G).cuda()
		self.randomlist = range(growth_rate*G)
		random.shuffle(self.randomlist)

		for ii in range(growth_rate*G):
			self.index[ii] = self.randomlist[ii]

		# self.channelpassindex = torch.LongTensor(in_planes - 2*growth_rate).cuda()
		# self.randomlist = range(growth_rate*G)

		FC_k = 1
		FC_pad = 0

		# self.conv_fc_1 = nn.Conv2d(growth_rate, growth_rate, kernel_size=FC_k, padding=FC_pad, groups=1, stride=1, bias=False)
		# self.conv_fc_2 = nn.Conv2d(growth_rate, growth_rate, kernel_size=FC_k, padding=FC_pad, groups=1, stride=1, bias=False)
		

		# self.m_fc_1 = nn.ReLU()
		# self.m_fc_2 = nn.ReLU()
		# # self.m = nn.ReLU()
		# self.bn_fc_1 = nn.BatchNorm2d(growth_rate)
		# self.bn_fc_2 = nn.BatchNorm2d(growth_rate)


	def forward(self, x):
		out = self.conv1(F.relu(self.bn1(x))) # in_ch -> 4GR
		out_dc = self.conv2(F.relu(self.bn2(out))) # 4GR -> GR
		out = self.conv1_prcn(F.relu(self.bn3(out_dc))) # GR -> G*GR
		# out = self.conv1(out) # in_ch -> G*in_ch
		out = out[:,self.index,:,:] # randomization
		out = self.transpool(out)  # G*GR  ->  GR

		# out = self.bn_fc_1(out)
		# out = self.m_fc_1(out)
		# out = self.conv_fc_1(out) # in_ch -> fc_ch
		# # print(out.cpu().size())

		# out = self.bn_fc_2(out)
		# out = self.m_fc_2(out)
		# out = self.conv_fc_2(out) # fc_ch -> fc_ch
		# print(out.cpu().size())
		# print(self.in_planes)
		# print(self.growth_rate)
		out = torch.cat((out, out_dc, x[:,:self.in_planes - self.growth_rate ,:,:]),1) # in_ch -> in_ch + GR      
		# print(out.cpu().size())
		# [out_prcn, out_dc, skip]
		return out

########################################################################################
########################################################################################
########################################################################################





class PRCN_v7_Bottleneck(nn.Module):
	def __init__(self, in_planes, growth_rate):
		super(PRCN_v7_Bottleneck, self).__init__()

		# growth rate needs to be even
		# growth_rate /=2
		self.in_planes = in_planes
		self.growth_rate = growth_rate
		interChannels = 4*growth_rate
		self.bn1 = nn.BatchNorm2d(in_planes)
		self.conv1 = nn.Conv2d(in_planes, interChannels, kernel_size=1,
							   bias=False, groups=1)
		self.bn2 = nn.BatchNorm2d(interChannels)
		self.conv2 = nn.Conv2d(interChannels, growth_rate, kernel_size=3,
							   padding=1, bias=False)

		# self.conv_pre = nn.Conv2d(in_planes, growth_rate, kernel_size=1, bias=False)
		G = 4
		# self.bn1 = nn.BatchNorm2d(in_planes)
		self.bn3 = nn.BatchNorm2d(growth_rate)
		self.conv1_prcn = nn.Conv2d(growth_rate, G*growth_rate, kernel_size=3, padding=1, groups=growth_rate, bias=False)
		# self.bn2 = nn.BatchNorm2d(4*growth_rate)
		# self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

		self.transpool = nn.MaxPool3d((G, 1, 1))
		self.index = torch.LongTensor(growth_rate*G).cuda()
		self.randomlist = range(growth_rate*G)
		random.shuffle(self.randomlist)

		for ii in range(growth_rate*G):
			self.index[ii] = self.randomlist[ii]


	def forward(self, x):
		# print(x.cpu().size())

		out = self.conv1(F.relu(self.bn1(x))) # in_ch -> 4GR
		# print(out.cpu().size())
		out_dc = self.conv2(F.relu(self.bn2(out))) # 4GR -> GR
		# print(out.cpu().size())
		out = self.conv1_prcn(F.relu(self.bn3(out_dc))) # GR -> G*GR
		# print(out.cpu().size())
		# out = self.conv1(out) # in_ch -> G*in_ch
		out = out[:,self.index,:,:] # randomization
		out = self.transpool(out)  # G*GR  ->  GR
		# out = torch.cat((out, out_dc, x[:,:self.in_planes - 2*self.growth_rate ,:,:]),1) # in_ch -> in_ch + GR      
		out = torch.cat((out, out_dc, x[:,:self.in_planes - 2*self.growth_rate ,:,:]),1) # in_ch -> in_ch     
		# print(out.cpu().size())
		# [out_prcn, out_dc, skip]
		return out

########################################################################################
########################################################################################
########################################################################################

class PRCN_v8_Bottleneck(nn.Module):
	def __init__(self, in_planes, growth_rate):
		super(PRCN_v8_Bottleneck, self).__init__()


		G = 2
		self.expansion=1
		# growth rate needs to be even
		# growth_rate /=2
		self.in_planes = in_planes
		self.growth_rate = growth_rate
		interChannels = growth_rate
		self.bn1 = nn.BatchNorm2d(growth_rate)
		self.conv1 = nn.Conv2d(G*growth_rate*self.expansion, growth_rate, kernel_size=1, bias=False, groups=growth_rate)
		self.bn2 = nn.BatchNorm2d(self.expansion*growth_rate*G)
		self.conv2 = nn.Conv2d(growth_rate, growth_rate, kernel_size=1, padding=0, bias=False)

		# self.conv_pre = nn.Conv2d(in_planes, growth_rate, kernel_size=1, bias=False)
		# self.bn1 = nn.BatchNorm2d(in_planes)
		self.bn3 = nn.BatchNorm2d(growth_rate)
		self.conv1_prcn = nn.Conv2d(growth_rate, G*growth_rate, kernel_size=3, padding=1, groups=growth_rate, bias=False)
		# self.bn2 = nn.BatchNorm2d(4*growth_rate)
		# self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

		self.transpool = nn.MaxPool3d((G, 1, 1))


		self.index_1 = torch.LongTensor(growth_rate*G*self.expansion).cuda()
		self.randomlist = list(range(growth_rate*G*self.expansion))
		random.shuffle(self.randomlist)
		for ii in list(range(growth_rate*G*self.expansion)):
			self.index_1[ii] = self.randomlist[ii]


		self.index_2 = torch.LongTensor(growth_rate*G*self.expansion).cuda()
		self.randomlist = list(range(growth_rate*G*self.expansion))
		random.shuffle(self.randomlist)
		for ii in list(range(growth_rate*G*self.expansion)):
			self.index_2[ii] = self.randomlist[ii]

		self.index_in = torch.LongTensor(growth_rate).cuda()
		self.index_out = torch.LongTensor(growth_rate).cuda()
		self.randomlist = list(range(in_planes))
		random.shuffle(self.randomlist)

		# for ii in list(range(growth_rate)):
			# self.index_in[ii] = self.randomlist[ii]

		self.index_in = self.randomlist[:self.growth_rate]
		self.index_out = self.randomlist[self.growth_rate:]

		self.bn4 = nn.BatchNorm2d(in_planes + growth_rate)

	def forward(self, x):
		# print(x.cpu().size())
		# print(self.index_in)

		# stream 1
		out_shuff = self.conv1_prcn(self.bn1(x[:,self.index_in,:,:])) # GR -> G*GR
		# out = out.repeat(1,self.expansion,1,1)  # G*GR -> G*GR*expansion
		out_shuff = out_shuff[:,self.index_1,:,:] # randomization
		# out = out[:,self.index_2,:,:] # randomization
		out_pool = self.transpool(out_shuff)  # G*GR*expansion  ->  GR*expansion
		out = self.conv1(F.relu(self.bn2(out_shuff))) # GR*expansion -> GR
		# print(out.cpu().size())
		out = self.conv2(F.relu(self.bn3(out))) # GR -> GR
		# print(out.cpu().size())
		# out = self.conv1(out) # in_ch -> G*in_ch
		# out = torch.cat((out, out_dc, x[:,:self.in_planes - 2*self.growth_rate ,:,:]),1) # in_ch -> in_ch + GR      
		out = torch.cat((out, out_pool, x[:,self.index_out,:,:]),1) # in_ch -> in_ch     
		# print(out.cpu().size())
		# [out_prcn, out_dc, skip]
		put = self.bn4(out)
		return out

########################################################################################
########################################################################################
########################################################################################


class PRCN_v10_Bottleneck(nn.Module):
	def __init__(self, in_planes, growth_rate):
		super(PRCN_v10_Bottleneck, self).__init__()

		#10.2
		G = 4
		self.expansion=1
		# growth rate needs to be even
		# growth_rate /=2
		self.in_planes = in_planes
		self.growth_rate = growth_rate
		interChannels = growth_rate
		self.bn1 = nn.BatchNorm2d(in_planes)
		self.conv1 = nn.Conv2d(G*growth_rate*self.expansion, growth_rate, kernel_size=1, bias=False, groups=growth_rate)
		self.bn2 = nn.BatchNorm2d(self.expansion*growth_rate*G)
		self.conv2 = nn.Conv2d(growth_rate, growth_rate, kernel_size=1, padding=0, bias=False)

		# self.conv_pre = nn.Conv2d(in_planes, growth_rate, kernel_size=1, bias=False)
		# self.bn1 = nn.BatchNorm2d(in_planes)
		self.bn3 = nn.BatchNorm2d(growth_rate)
		self.conv1_prcn = nn.Conv2d(in_planes, G*growth_rate, kernel_size=3, padding=1, groups=growth_rate, bias=False)
		# self.bn2 = nn.BatchNorm2d(4*growth_rate)
		# self.conv2 = nn.Conv2d(4*growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

		self.transpool = nn.MaxPool3d((G, 1, 1))


		self.index_1 = torch.LongTensor(growth_rate*G*self.expansion).cuda()
		self.randomlist = list(range(growth_rate*G*self.expansion))
		random.shuffle(self.randomlist)
		for ii in list(range(growth_rate*G*self.expansion)):
			self.index_1[ii] = self.randomlist[ii]


		# self.index_2 = torch.LongTensor(growth_rate*G*self.expansion).cuda()
		# self.randomlist = list(range(growth_rate*G*self.expansion))
		# random.shuffle(self.randomlist)
		# for ii in list(range(growth_rate*G*self.expansion)):
		# 	self.index_2[ii] = self.randomlist[ii]

		self.index_in = torch.LongTensor(in_planes).cuda()
		# self.index_out = torch.LongTensor(growth_rate).cuda()
		self.randomlist = list(range(in_planes))
		random.shuffle(self.randomlist)

		# # for ii in list(range(growth_rate)):
		# 	# self.index_in[ii] = self.randomlist[ii]

		# self.index_in = self.randomlist[:self.growth_rate]
		self.index_in = self.randomlist
		# self.index_out = self.randomlist[self.growth_rate:]

		self.bn4 = nn.BatchNorm2d(in_planes + growth_rate)

	def forward(self, x):
		# print(x.cpu().size())
		# print(self.index_in)

		# stream 1
		out = self.conv1_prcn(self.bn1(x[:,self.index_in,:,:])) # GR -> G*GR
		# out = out.repeat(1,self.expansion,1,1)  # G*GR -> G*GR*expansion
		out = out[:,self.index_1,:,:] # randomization
		# print(out.cpu().size())
		# out = out[:,self.index_2,:,:] # randomization
		# out_pool = self.transpool(out_shuff)  # G*GR*expansion  ->  GR*expansion
		out = self.conv1(F.relu(self.bn2(out))) # G*GR*expansion -> GR
		# print(out.cpu().size())
		out = self.conv2(F.relu(self.bn3(out))) # GR -> GR
		# print(out.cpu().size())
		# out = self.conv1(out) # in_ch -> G*in_ch
		# out = torch.cat((out, out_dc, x[:,:self.in_planes - 2*self.growth_rate ,:,:]),1) # in_ch -> in_ch + GR      
		# out = torch.cat((out, out_pool, x[:,self.index_out,:,:]),1) # in_ch -> in_ch     
		out = torch.cat((out, x),1) # in_ch -> in_ch   + GR  
		# print(out.cpu().size())
		# [out_prcn, out_dc, skip]
		put = self.bn4(out)
		return out

########################################################################################
########################################################################################
########################################################################################
########################################################################################


class PRCN_v11_Bottleneck(nn.Module):
	def __init__(self, in_planes, growth_rate):
		super(PRCN_v11_Bottleneck, self).__init__()


		G = 4
		#11.1
		nChannels = in_planes

		# growth rate needs to be even
		interChannels = 4*growth_rate
		self.bn1 = nn.BatchNorm2d(nChannels)
		self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1, bias=False, groups=growth_rate)
		self.bn2 = nn.BatchNorm2d(interChannels)
		self.conv2 = nn.Conv2d(interChannels, G*growth_rate, kernel_size=3, padding=1, bias=False, groups=growth_rate)
		

		self.index_1 = torch.LongTensor(G*growth_rate).cuda()
		self.randomlist = list(range(G*growth_rate))
		random.shuffle(self.randomlist)
		for ii in list(range(G*growth_rate)):
			self.index_1[ii] = self.randomlist[ii]
		# self.drop1 = nn.Dropout2d(p=0.2)
		# self.drop2 = nn.Dropout2d(p=0.2)


		# self.index_in = torch.LongTensor(in_planes).cuda()
		# # self.index_out = torch.LongTensor(growth_rate).cuda()
		# self.randomlist = list(range(in_planes))
		# random.shuffle(self.randomlist)

		# # for ii in list(range(growth_rate)):
		# 	# self.index_in[ii] = self.randomlist[ii]

		# self.index_in = self.randomlist[:self.growth_rate]
		self.index_in = self.randomlist

		self.bn4 = nn.BatchNorm2d(in_planes + growth_rate)


	def forward(self, x):
		out = self.conv1(F.relu(self.bn1(x)))
		# out = self.drop1(out)

		out = self.conv2(F.relu(self.bn2(out)))
		# out = self.drop2(out)
		out = out[:,self.index_1,:,:] # randomization

		out = torch.cat((x, out), 1)
		# print(out.cpu().size())
		# [out_prcn, out_dc, skip]
		# put = self.bn4(out)
		return out

########################################################################################
########################################################################################
########################################################################################


class Bottleneck(nn.Module):
	def __init__(self, nChannels, growthRate):
		super(Bottleneck, self).__init__()
		interChannels = 4*growthRate
		self.bn1 = nn.BatchNorm2d(nChannels)
		self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
							   bias=False)
		self.bn2 = nn.BatchNorm2d(interChannels)
		self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
							   padding=1, bias=False)

		# self.drop1 = nn.Dropout2d(p=0.2)
		# self.drop2 = nn.Dropout2d(p=0.2)

	def forward(self, x):
		out = self.conv1(F.relu(self.bn1(x)))
		# out = self.drop1(out)

		out = self.conv2(F.relu(self.bn2(out)))
		# out = self.drop2(out)
		
		out = torch.cat((x, out), 1)
		return out

########################################################################################
########################################################################################
########################################################################################

class PRCN_layer(nn.Module):
	def __init__(self, nChannels, outchannels, kernel_size, padding=0, bias =False):
		super(PRCN_layer, self).__init__()


		self.maxpool_size = 2
		self.avgpool_size = nChannels//self.maxpool_size
		self.expansion = outchannels*nChannels
		self.conv1 = nn.Conv2d(nChannels, outchannels*nChannels, kernel_size=kernel_size, bias=bias, groups=nChannels, padding=padding)

		self.transpool1 = nn.MaxPool3d((self.maxpool_size, 1, 1))
		self.transpool2 = nn.AvgPool3d((self.avgpool_size, 1, 1))

		self.index = torch.LongTensor(self.expansion).cuda()
		self.randomlist = list(range(self.expansion))
		random.shuffle(self.randomlist)

		for ii in range(self.expansion):
			self.index[ii] = self.randomlist[ii]


	def forward(self, x):
		out = self.conv1(x) # nChannels -> outchannels*nChannels
		# out = self.drop1(out)
		# out = out.repeat(1,self.expansion,1,1)  # G*in_ch*out_ch -> G*in_ch*out_ch*expansion
		out = out[:,self.index,:,:] # randomization
		out = self.transpool1(out) # outchannels*nChannels -> outchannels
		out = self.transpool2(out) # outchannels*nChannels -> outchannels
		return out

########################################################################################
########################################################################################
########################################################################################


class PRCN_v12_Bottleneck(nn.Module):
	def __init__(self, nChannels, growthRate):
		super(PRCN_v12_Bottleneck, self).__init__()
		interChannels = 4*growthRate
		self.bn1 = nn.BatchNorm2d(nChannels)
		# self.conv1 = PRCN_layer(nChannels, interChannels, kernel_size=1, bias=False)
		self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1, bias=False)
		self.bn2 = nn.BatchNorm2d(interChannels)
		self.conv2 = PRCN_layer(interChannels, growthRate, kernel_size=3, padding=1, bias=False)

		# self.drop1 = nn.Dropout2d(p=0.2)
		# self.drop2 = nn.Dropout2d(p=0.2)

	def forward(self, x):
		out = self.conv1(F.relu(self.bn1(x)))
		# out = self.drop1(out)

		out = self.conv2(F.relu(self.bn2(out)))
		# out = self.drop2(out)
		
		out = torch.cat((x, out), 1)
		return out

########################################################################################
########################################################################################
########################################################################################

class DenseNet_PRCNv7(nn.Module):
	def __init__(self, growthRate, depth, reduction, nClasses, bottleneck):
		super(DenseNet_PRCNv7, self).__init__()

		# nDenseBlocks = (depth-4) // 3
		# if bottleneck:
			# nDenseBlocks //= 2
		# config = [32, 32, 32]
		# config = [16, 16, 16]
		# config = [12, 12, 12]
		# config = [9, 9, 9]
		# config = [8, 8, 8]
		# config = [7, 7, 7]
		# config = [6, 6, 6]
		# config = [4, 4, 4]
		# config = [3, 3, 3]
		# config = [2, 3, 4]

		import pdb; pdb.set_trace()
		n = 2
		growthRate = 12
		
		Gmul = 2
		config = [n,n,n]


		nChannels = Gmul*growthRate

		block_type = 9 # 0 is bottleneck, 1 is PRCN, 2 is PRCNv4, 3 is PRCNv5
					   # 4 is PRCNv6, 5 is PRCNv7, 6 is PRCNv8, 7 is PRCNv10
					   # 8 is PRCNv11, 9 is PRCNv12

		self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1,
							   bias=False)
		nDenseBlocks = config[0]
		self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, block_type)
		nChannels += nDenseBlocks*growthRate
		nOutChannels = int(math.floor(nChannels*reduction))
		self.trans1 = Transition(nChannels, nOutChannels)

		nChannels = nOutChannels
		nDenseBlocks = config[1]
		self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, block_type)
		nChannels += nDenseBlocks*growthRate
		nOutChannels = int(math.floor(nChannels*reduction))
		self.trans2 = Transition(nChannels, nOutChannels)

		nChannels = nOutChannels
		nDenseBlocks = config[2]
		self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, block_type)
		nChannels += nDenseBlocks*growthRate

		self.bn1 = nn.BatchNorm2d(nChannels)
		self.fc = nn.Linear(nChannels, nClasses)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()
			elif isinstance(m, nn.Linear):
				m.bias.data.zero_()

	def _make_dense(self, nChannels, growthRate, nDenseBlocks, block_type):
		layers = []
		for i in range(int(nDenseBlocks)):
			if block_type == 0:
				layers.append(Bottleneck(nChannels, growthRate))
				print('Bottleneck')
			elif block_type == 1:
				layers.append(PRCN_v3_Bottleneck(nChannels, growthRate))				
				print('PRCN_v3_Bottleneck')
			elif block_type == 2:
				layers.append(PRCN_v4_Bottleneck(nChannels, growthRate))				
				print('PRCN_v4_Bottleneck')
			elif block_type == 3:
				layers.append(PRCN_v5_Bottleneck(nChannels, growthRate))				
				print('PRCN_v5_Bottleneck')
			elif block_type == 4:
				layers.append(PRCN_v6_Bottleneck(nChannels, growthRate))				
				print('PRCN_v6_Bottleneck')
			elif block_type == 5:
				layers.append(PRCN_v7_Bottleneck(nChannels, growthRate))				
				print('PRCN_v7_Bottleneck')	
			elif block_type == 6:
				layers.append(PRCN_v8_Bottleneck(nChannels, growthRate))				
				print('PRCN_v8_Bottleneck')							
			elif block_type == 7:
				layers.append(PRCN_v10_Bottleneck(nChannels, growthRate))				
				print('PRCN_v10_Bottleneck')	
			elif block_type == 8:
				layers.append(PRCN_v11_Bottleneck(nChannels, growthRate))				
				print('PRCN_v11_Bottleneck')
			elif block_type == 9:
				layers.append(PRCN_v12_Bottleneck(nChannels, growthRate))				
				print('PRCN_v12_Bottleneck')
			else:
				layers.append(SingleLayer(nChannels, growthRate))
			nChannels += growthRate
		return nn.Sequential(*layers)

	def forward(self, x):
		out = self.conv1(x)
		out = self.dense1(out)
		# print(out.cpu().size())
		out = self.trans1(out)
		# print(out.cpu().size())
		out = self.trans2(self.dense2(out))
		out = self.dense3(out)
		# print(out.cpu().size())
		out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))
		out = F.log_softmax(self.fc(out))
		return out

def test(): 
	# CIFAR - 100
	# model = DenseNet_PRCNv7(
	# 	growthRate=12, 
	# 	depth=100, 
	# 	reduction=1.0, 
	# 	bottleneck=True,
	# 	nClasses=100
	# )
	# CIFAR - 10
	model = DenseNet_PRCNv7(
		growthRate=12, 
		depth=100, 
		reduction=0.5, 
		bottleneck=True,
		nClasses=10
	)

	# Calculate number of parameters 
	#from thop import profile 
	x_ = torch.randn(2, 3, 32, 32)
	op = model(x_)
	print(op.shape)
	#macs, params = profile(model, inputs=(input, ))
	#print("params: {} flops: {}".format(params, macs))

if __name__ == "__main__":
	test()