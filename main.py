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
from tensorboardX import SummaryWriter

##  新增的一个import
from input_pipeline import DatasetFolder

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


#参数设定
Load_Pre_Trained = True
Training_dir = '/media/zhineng/Data/M/aligned_imgs/'
_batch_size = 20
_seq_num = 32
_classnum=1400
Save_Model = '/media/zhineng/Data/M/Models/cnn_cnn_'
Print_Labels = False


#get file number
filenum = 0
for speaker in os.listdir(Training_dir):
	for video in os.listdir(Training_dir+'/'+speaker):
		filenum+=1
epoch_filenum = math.floor(filenum/_batch_size)*_batch_size
print('File Number is %d, Epoch File Number is %d'%(filenum,epoch_filenum))

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

writer = SummaryWriter('cnn_cnn')
for epoch in range(5000):  # loop over the dataset multiple times

	running_loss = 0.0
	correct = 0
	total = 0
	for i, data in enumerate(trainloader, 0):

		if i>(math.floor(filenum/_batch_size)-1): continue
		
		# get the inputs
		inputs, labels = data
		inputs, labels = inputs.to(device), labels.to(device)

		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		outputs = net(inputs)
		_, predicted = torch.max(outputs, 1)
		
		# print(outputs)

		
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		# print statistics & draw on tensorboard
		running_loss += loss.item()
		total += labels.size(0)
		correct += (predicted==labels).sum().item()
		if i % 5 == 4:    # print every 5 mini-batches
			print('[%d, %5d] loss: %.3f' %
				  (epoch + 1, i + 1, running_loss / 5))
			if Print_Labels:
				print('predicts')
				print(predicted)
				print('labels')
				print(labels)
			accuracy = correct/total
			print('Accuray: %d %%'%(100*accuracy))
			writer.add_scalar('train/loss',running_loss / 5, epoch*epoch_filenum + 5*(i+1))
			writer.add_scalar('train/accuracy',accuracy,epoch*epoch_filenum + 5*(i+1))
			running_loss = 0.0
			correct = 0
			total = 0

	if epoch%200 == 199:
		torch.save(net.state_dict(),Save_Model+ str(epoch+1) +'.pth')

writer.close()
print('Finished Training')




















