from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import net_sphere
import cnn_cnn
import os
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision
import math

##  新增的一个import
from input_pipeline import DatasetFolder

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


#参数设定
Load_Pre_Trained = True
Training_dir = '/media/zhineng/Data/M/aligned_imgs_test/'
_batch_size = 20
_seq_num = 32
_classnum=194



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assume that we are on a CUDA machine, then this should print a CUDA device:
print(device)



net = getattr(cnn_cnn,'cnn_cnn_')(batch_size=_batch_size,seq_num=_seq_num,classnum=_classnum,feature=False)
net_dict = net.state_dict()

if Load_Pre_Trained:
	load_dict = torch.load(os.getcwd()+'/sphere20a_20171020.pth')
	pretrained_dict = {k: v for k, v in load_dict.items() if k in net_dict}
	net_dict.update(pretrained_dict)
	net.load_state_dict(net_dict)
	# print(net.state_dict())
	# print(net.parameters()) 

net.to(device)


count = 0
para_optim = []
for k in net.children():
	count = count + 1
	if count < 42:
		for paras in k.parameters():
			paras.requires_grad = False
	else:
		for paras in k.parameters():
			para_optim.append(paras)


# transform dictionary
# ToTensor 是必选项, 否则格式不支持
transform = transforms.Compose(
	[transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  



trainset = DatasetFolder(Training_dir,transform=transform,extensions='.jpg',seq_num=_seq_num)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=_batch_size,
										  shuffle=True, num_workers=1)


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(para_optim,lr=0.00001,momentum=0.9)


for epoch in range(1000):  # loop over the dataset multiple times

	running_loss = 0.0
	for i, data in enumerate(trainloader, 0):

		if i>(math.floor(435/_batch_size)-1): continue
		
		# get the inputs
		inputs, labels = data
		

		inputs, labels = inputs.to(device), labels.to(device)

		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		outputs = net(inputs)
		_, predicted = torch.max(outputs, 1)
		
		# print(outputs)

		# nn.LogSoftMax
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		# print statistics
		running_loss += loss.item()
		if i % 5 == 4:    # print every 20 mini-batches
			print('[%d, %5d] loss: %.3f' %
				  (epoch + 1, i + 1, running_loss / 5))
			print('predicts')
			print(predicted)
			print('labels')
			print(labels)
			running_loss = 0.0

print('Finished Training')




















