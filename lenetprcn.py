'''LeNet in PyTorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
# import numpy as np

# from pool_ops import FusedMultiPool





################################################################################
################################################################################
################################################################################






################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################

class NPTN_vanquish(nn.Module):
	expansion = 1

	def __init__(self,  in_ch, out_ch, G, k, pad, stride ):
		super(NPTN_vanquish, self ).__init__()
		self.conv1 = nn.Conv2d(in_ch, G*in_ch*out_ch, kernel_size=k, padding=pad, groups=in_ch, stride=stride, bias=True)
		self.transpool = nn.MaxPool3d((G, 1, 1))
		self.chanpool = nn.AvgPool3d((in_ch, 1, 1))
		# self.chanpool = nn.MaxPool3d((in_ch, 1, 1),stride=(1,1,1), dilation=(out_ch,1,1))
		# self.chanpool = nn.AvgPool3d((1, 1, 1))
		# self.bn1 = nn.BatchNorm2d(G*in_ch*out_ch)
		# self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=pool, stride=pool, groups=out_ch)
		# self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=pool, stride=pool, groups=out_ch)
		self.bn2 = nn.BatchNorm2d(out_ch)
		# self.m1 = nn.PReLU()
		self.m2 = nn.PReLU()
		self.out_ch = out_ch
		self.G = G
		self.index = torch.LongTensor(in_ch*out_ch).cuda()

		index = 0
		for ii in range(in_ch):
			for jj in range(out_ch):
				self.index[ii + jj*in_ch] = index
				index+=1
		# print(self.index)



	def forward(self, x):

		out = self.conv1(x) 
		# out = self.bn1(out)
		# out = self.m1(out)
		# out = self.drop(out)
		# print(out.data.size())

		out = self.transpool(out)
		out = out[:,self.index,:,:]
		# print(out.data.size())
		out = self.chanpool(out)


		# print(out.data.size())
		# out = self.conv2(out) # in_ch*out_ch*expansion  ->  out_ch
		# out = self.m2(out)
		# out = self.conv3(out) # out_ch  ->  out_ch
		# out = torch.sum(out, 1, keepdim=True)
		# out = self.conv2(out) # diff from db9 ori


		# print(out.data.size())
		# out += residual
		return out



class NPTN_vanquish_ablation(nn.Module):
	expansion = 1

	def __init__(self,  in_ch, out_ch, G, k, pad, stride ):
		super(NPTN_vanquish_ablation, self ).__init__()
		self.conv1 = nn.Conv2d(in_ch, G*in_ch*out_ch, kernel_size=k, padding=pad, groups=in_ch, stride=stride, bias=True)
		self.transpool = nn.MaxPool3d((G, 1, 1))
		self.chanpool = nn.AvgPool3d((in_ch, 1, 1))
		# self.chanpool = nn.MaxPool3d((in_ch, 1, 1),stride=(1,1,1), dilation=(out_ch,1,1))
		# self.chanpool = nn.AvgPool3d((1, 1, 1))
		# self.bn1 = nn.BatchNorm2d(G*in_ch*out_ch)
		# self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=pool, stride=pool, groups=out_ch)
		# self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=pool, stride=pool, groups=out_ch)
		self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, groups=1)
		# self.conv2 = nn.Conv2d(out_ch*self.expansion, out_ch, kernel_size=1, stride=1, groups=1)
		self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, groups=1)

		self.bn2 = nn.BatchNorm2d(out_ch)
		# self.m1 = nn.PReLU()
		self.m2 = nn.PReLU()
		self.out_ch = out_ch
		self.G = G
		self.index = torch.LongTensor(in_ch*out_ch).cuda()

		index = 0
		for ii in range(in_ch):
			for jj in range(out_ch):
				self.index[ii + jj*in_ch] = index
				index+=1
		# print(self.index)



	def forward(self, x):

		out = self.conv1(x) 
		# out = self.bn1(out)
		# out = self.m1(out)
		# out = self.drop(out)
		# print(out.data.size())

		out = self.transpool(out)
		out = out[:,self.index,:,:]
		# print(out.data.size())
		out = self.chanpool(out)


		# print(out.data.size())
		# out = self.conv2(out) # in_ch*out_ch*expansion  ->  out_ch
		# out = self.m2(out)
		# out = self.conv3(out) # out_ch  ->  out_ch
		# out = torch.sum(out, 1, keepdim=True)
		# out = self.conv2(out) # diff from db9 ori
		out = self.conv2(out) # in_ch*out_ch*expansion  ->  out_ch
		out = self.m2(out)
		out = self.conv3(out) # out_ch  ->  out_ch

		# print(out.data.size())
		# out += residual
		return out

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################


################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
class RPTNv1(nn.Module):

	def __init__(self,  in_ch, out_ch, G, exp, k, pad, stride ):
		super(RPTNv1, self ).__init__()
		self.conv1 = nn.Conv2d(in_ch, G*in_ch*out_ch, kernel_size=k, padding=pad, groups=in_ch, stride=stride, bias=True)
		# self.transpool = nn.MaxPool3d((G, 1, 1))
		self.transpool = nn.AvgPool3d((G, 1, 1))

		self.expansion = exp


		self.conv2 = nn.Conv2d(in_ch * out_ch*self.expansion, out_ch, kernel_size=1, stride=1, groups=1)
		# self.conv2 = nn.Conv2d(out_ch*self.expansion, out_ch, kernel_size=1, stride=1, groups=1)
		self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, groups=1)
		# self.conv4 = nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, groups=1)
		# self.conv5 = nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, groups=1)
		# self.conv6 = nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, groups=1)



		self.bn2 = nn.BatchNorm2d(out_ch)

		self.m2 = nn.PReLU()
		self.m3 = nn.PReLU()
		# self.m4 = nn.PReLU()
		# self.m5 = nn.PReLU()
		# self.m6 = nn.PReLU()


		self.out_ch = out_ch
		self.G = G
		self.index = torch.LongTensor(in_ch*out_ch*G*self.expansion).cuda()
		self.randomlist = list(range(in_ch*out_ch*G*self.expansion))
		random.shuffle(self.randomlist)

		for ii in range(in_ch*out_ch*G*self.expansion):
			self.index[ii] = self.randomlist[ii]

	def forward(self, x):

		out = self.conv1(x) # in_ch -> G*in_ch*out_ch
		# out = out.repeat(1,self.expansion,1,1)  # G*in_ch*out_ch -> G*in_ch*out_ch*expansion
		out = out[:,self.index,:,:] # randomization


		out = self.transpool(out)  # G*in_ch*out_ch*expansion  ->  in_ch*out_ch*expansion

		out = self.conv2(out) # in_ch*out_ch*expansion  ->  out_ch
		out = self.m2(out)
		out = self.conv3(out) # out_ch  ->  out_ch


		return out

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
class baseline(nn.Module):

	def __init__(self,  in_ch, out_ch, k, pad, stride ):
		super(baseline, self ).__init__()
		# self.conv1 = nn.Conv2d(in_ch, G*in_ch*out_ch, kernel_size=k, padding=pad, groups=in_ch, stride=stride, bias=True)
		self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=pad, bias=True)
		self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=1, padding=0, bias=True)
		self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=1, padding=0, bias=True)

		self.m2 = nn.PReLU()
		self.m3 = nn.PReLU()


	def forward(self, x):

		out = self.conv1(x) # in_ch -> G*in_ch*out_ch
		out = self.m2(out)
		out = self.conv2(out) # in_ch*out_ch*expansion  ->  out_ch
		out = self.m3(out)
		out = self.conv3(out) # out_ch  ->  out_ch


		return out

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################





class PRCN_layer(nn.Module):
	def __init__(self, nChannels, outchannels, G, exp, kernel_size, padding, stride):
		super(PRCN_layer, self).__init__()

		self.exp = exp
		self.G = G
		self.maxpool_size = G
		self.avgpool_size = int((nChannels*exp)/G)
		self.expansion = outchannels*nChannels*self.exp
		self.conv1 = nn.Conv2d(nChannels, outchannels*nChannels, kernel_size=kernel_size, groups=nChannels, padding=padding, bias=True)

		self.transpool1 = nn.MaxPool3d((self.maxpool_size, 1, 1))
		self.transpool2 = nn.AvgPool3d((self.avgpool_size, 1, 1))

		self.index = torch.LongTensor(self.expansion).cuda()
		self.randomlist = list(range(self.expansion))
		random.shuffle(self.randomlist)

		for ii in range(self.expansion):
			self.index[ii] = self.randomlist[ii]


	def forward(self, x):
		import pdb; pdb.set_trace()
		# print(x.data.size())
		out = self.conv1(x) # nChannels -> outchannels*nChannels
		# print(out.data.size())
		# out = self.drop1(out)
		out = out.repeat(1,self.exp,1,1)  # outchannels*nChannels-> exp*outchannels*nChannels
		out = out[:,self.index,:,:] # randomization
		# print(out.data.size())
		out = self.transpool1(out) # exp*outchannels*nChannels -> exp*outchannels*nChannels/G
		# print(out.data.size())
		out = self.transpool2(out) # exp*outchannels*nChannels/(G*meanpool) -> outchannels
		# print(out.data.size())

		return out
# notes: feel like we should have more expansion or reuse, but lesser max pool say 2 and more mean poolrest of it to maintain
# o/p ch
########################################################################################
########################################################################################
########################################################################################


class PRCN_layer_versionG(nn.Module):
	def __init__(self, nChannels, outchannels, G, exp, CMP, kernel_size, padding, stride):
		super(PRCN_layer_versionG, self).__init__()

		self.exp = exp
		self.G = G
		self.maxpool_size = CMP
		self.avgpool_size = int((nChannels*self.exp*self.G)/(self.maxpool_size*outchannels))
		self.expansion = self.G*nChannels*self.exp
		self.conv1 = nn.Conv2d(nChannels, self.G*nChannels, kernel_size=kernel_size, groups=nChannels, padding=padding, bias=True)

		self.transpool1 = nn.MaxPool3d((self.maxpool_size, 1, 1))
		self.transpool2 = nn.AvgPool3d((self.avgpool_size, 1, 1))

		self.index = torch.LongTensor(self.expansion).cuda()
		self.randomlist = list(range(self.expansion))
		random.shuffle(self.randomlist)

		for ii in range(self.expansion):
			self.index[ii] = self.randomlist[ii]
		self.prelu1 = nn.PReLU()
		self.prelu2 = nn.PReLU()

		# print(nChannels*G*exp/self.maxpool_size)
		self.fc1 = nn.Conv2d(int(nChannels*G*exp/self.maxpool_size), outchannels, kernel_size=1, padding=0, bias=True)
		self.fc2 = nn.Conv2d(outchannels, outchannels, kernel_size=1, padding=0, bias=True)



	def forward(self, x):
		# print(x.data.size())
		out = self.conv1(x) # nChannels -> G*nChannels
		# print(out.data.size())
		# out = self.drop1(out)
		out = out.repeat(1,self.exp,1,1)  # G*nChannels-> exp*G*nChannels
		out = out[:,self.index,:,:] # randomization
		# print(out.data.size())
		out = self.transpool1(out) # exp*G*nChannels -> exp*nChannels*G/maxpool
		# print(out.data.size())
		# out = self.transpool2(out) # exp*nChannels*G/(maxpool*meanpool) -> outchannels


		out = self.prelu1(out)
		out = self.fc1(out)
		out = self.prelu2(out)
		out = self.fc2(out)



		return out
# notes: feel like we should have more expansion or reuse, but lesser max pool say 2 and more mean poolrest of it to maintain
# o/p ch
########################################################################################
########################################################################################
########################################################################################
class PRCN_fast(nn.Module):
	def __init__(self, nChannels, outchannels, G, exp, kernel_size, padding, stride):
		super(PRCN_fast, self).__init__()
		self.nChannels = nChannels
		self.outChannels = outchannels
		self.G = G 
		self.exp = exp
		self.channel_idx_sets = self.create_channel_idx_sets()
		self.conv1 = nn.Conv2d(nChannels, self.outChannels*nChannels, kernel_size=kernel_size, groups=nChannels, padding=padding, bias=True)
		self.fused_pool = FusedMultiPool(self.channel_idx_sets)
		self.avgpool_size = int((nChannels*exp)/G)
		self.transpool2 = nn.AvgPool3d((self.avgpool_size, 1, 1))

	
	def create_channel_idx_sets(self,):

		randomList = list(range(self.nChannels * self.outChannels * self.exp))
		random.shuffle(randomList)
		randomList = np.array(randomList)
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
########################################################################################
########################################################################################
########################################################################################

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
class RPTNv1_ablation(nn.Module):

	def __init__(self,  in_ch, out_ch, G, exp, k, pad, stride ):
		super(RPTNv1_ablation, self ).__init__()
		self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=pad, stride=stride, bias=True)
		# self.transpool = nn.MaxPool3d((G, 1, 1))
		self.transpool = nn.AvgPool3d((G, 1, 1))

		self.expansion = exp


		# self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, groups=1)
		# self.conv2 = nn.Conv2d(out_ch*self.expansion, out_ch, kernel_size=1, stride=1, groups=1)
		# self.conv3 = nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, groups=1)
		# self.conv4 = nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, groups=1)
		# self.conv5 = nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, groups=1)
		# self.conv6 = nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, groups=1)



		self.bn2 = nn.BatchNorm2d(out_ch)

		self.m2 = nn.PReLU()
		self.m3 = nn.PReLU()
		# self.m4 = nn.PReLU()
		# self.m5 = nn.PReLU()
		# self.m6 = nn.PReLU()


		self.out_ch = out_ch
		self.G = G
		self.index = torch.LongTensor(out_ch).cuda()
		self.randomlist = list(range(out_ch))
		random.shuffle(self.randomlist)

		for ii in range(out_ch):
			self.index[ii] = self.randomlist[ii]

	def forward(self, x):

		out = self.conv1(x) # in_ch -> G*in_ch*out_ch
		# out = out.repeat(1,self.expansion,1,1)  # G*in_ch*out_ch -> G*in_ch*out_ch*expansion
		out = out[:,self.index,:,:] # randomization


		# out = self.transpool(out)  # G*in_ch*out_ch*expansion  ->  in_ch*out_ch*expansion

		# out = self.conv2(out) # in_ch*out_ch*expansion  ->  out_ch
		# out = self.m2(out)
		# out = self.conv3(out) # out_ch  ->  out_ch


		return out

################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
class LeNetNPTN(nn.Module):
	def __init__(self, G, exp, maxpool_size, conv_channels):
		super(LeNetNPTN, self).__init__()
		self.channels = conv_channels #32
		self.G = G
		self.exp = exp
		self.CMP = maxpool_size
		self.fc1 = nn.Linear(16*3*3, 10)

		self.bn1 = nn.BatchNorm2d(self.channels)
		# self.bn2 = nn.BatchNorm2d(self.channels)
		self.bn3 = nn.BatchNorm2d(16)
		self.m1 = nn.PReLU()
		# self.m2 = nn.PReLU()
		self.m3 = nn.PReLU()



################
		# self.nptn1 = NPTN_vanquish(in_ch=1, out_ch=self.channels, G=self.G, k=5, pad=2, stride=1)
		# # # self.nptn2 = NPTN_vanquish(in_ch=self.channels, out_ch=self.channels, G=self.G, k=5, pad=1, stride=1)
		# self.nptn3 = NPTN_vanquish(in_ch=self.channels, out_ch=16, G=self.G, k=5, pad=2, stride=1)

		# self.nptn1 = NPTN_conv2(in_ch=3, out_ch=self.channels, G=self.G, k=5, pad=0, stride=1)
		# self.nptn2 = NPTN_vanquish(in_ch=self.channels, out_ch=self.channels, G=self.G, k=5, pad=1, stride=1)
		# self.nptn3 = NPTN_conv2(in_ch=self.channels, out_ch=16, G=self.G, k=5, pad=0, stride=1)
# 
		# self.nptn1 = NPTN_vanquish(in_ch=1, out_ch=self.channels, G=self.G, k=5, pad=2, stride=1)
		# # self.nptn2 = NPTN_vanquish(in_ch=self.channels, out_ch=self.channels, G=self.G, k=5, pad=0, stride=1)
		# self.nptn3 = NPTN_vanquish(in_ch=self.channels, out_ch=16, G=self.G, k=5, pad=2, stride=1)

		# self.nptn1 = NPTN_vanquish_ablation(in_ch=1, out_ch=self.channels, G=self.G, k=5, pad=2, stride=1)
		# # self.nptn2 = NPTN_vanquish(in_ch=self.channels, out_ch=self.channels, G=self.G, k=5, pad=0, stride=1)
		# self.nptn3 = NPTN_vanquish_ablation(in_ch=self.channels, out_ch=16, G=self.G, k=5, pad=2, stride=1)


		# self.nptn1 = RandomRouteTN6(in_ch=3, out_ch=self.channels, G=self.G, k=5, pad=0, stride=1)
		# self.nptn3 = RandomRouteTN6(in_ch=self.channels, out_ch=16, G=self.G, k=5, pad=0, stride=1)


		# self.nptn1 = RPTNv1_ablation(in_ch=1, out_ch=self.channels, G=self.G, exp = self.exp, k=5, pad=2, stride=1)
		# # self.nptn2 = RPTNv1(in_ch=self.channels, out_ch=self.channels, G=self.G, k=5, pad=0, stride=1)
		# self.nptn3 = RPTNv1_ablation(in_ch=self.channels, out_ch=16, G=self.G, exp = self.exp, k=5, pad=2, stride=1)



		self.nptn1 = PRCN_layer(nChannels=1, outchannels=self.channels, G=self.CMP, exp = self.exp, kernel_size=5, padding=2, stride=1)
		self.nptn3 = PRCN_layer(nChannels=self.channels, outchannels=16, G=self.CMP, exp = self.exp, kernel_size=5, padding=2, stride=1)




		# self.nptn1 = PRCN_fast(nChannels=3, outchannels=self.channels, G=self.G, exp = self.exp, kernel_size=5, padding=2, stride=1)
		# self.nptn3 = PRCN_fast(nChannels=self.channels, outchannels=16, G=self.G, exp = self.exp, kernel_size=5, padding=2, stride=1)

		# self.nptn1 = baseline(1, self.channels, k=5, pad=2, stride=1)
		# self.nptn3 = baseline(self.channels, 16, k=5, pad=2, stride=1)

		# self.nptn1 = nn.Conv2d(1, self.channels, kernel_size=5, padding=2)
		# self.nptn3 = nn.Conv2d(self.channels, 16, kernel_size=5, padding=2)	
		# self.nptn1 = PRCN_layer_versionG(nChannels=1, outchannels=self.channels, G=self.G, exp = self.exp, CMP=self.CMP, kernel_size=5, padding=2, stride=1)
		# self.nptn3 = PRCN_layer_versionG(nChannels=self.channels, outchannels=16, G=self.G, exp = self.exp, CMP=self.CMP,  kernel_size=5, padding=2, stride=1)




	def forward(self, x):

		# print(x.data.size())

		# out = self.conv1(x)
		out = self.nptn1(x)
		# print(out.data.size())

		out = self.bn1(out)
		out = self.m1(out)
		out = F.max_pool2d(out, 3)

########################
		# out = self.conv2(out)
		# out = self.nptn2(out)
		# out = self.bn2(out)
		# out = self.m2(out)
		# out = F.max_pool2d(out, 2)


		# out = self.conv3(out)
		out = self.nptn3(out)
		out = self.bn3(out)
		out = self.m3(out)
		out = F.max_pool2d(out, 3)
		# print(out.data.size())

		out = out.view(out.size(0), -1)
		# print(out.data.size())
		out = self.fc1(out)
		# a=f
		return out

def test(): 
	import pdb; pdb.set_trace()
	net = LeNetNPTN(G=2, exp=1, maxpool_size=2, conv_channels=18)
	x = torch.randn(2,1,28,28)
	y = net(x)
if __name__ == "__main__": 
	test()

