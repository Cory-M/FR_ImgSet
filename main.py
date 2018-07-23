from __future__ import print_function, division
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import net_sphere, cnn_cnn, multi_channel
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

Model = 'multi_channel'  #choose from {'cnn_cnn','multi_channel'}

#参数设定
if Model == 'cnn_cnn':
	Load_Pre_Trained = True
	Print_Labels = False
	Fix_preload_para = True

	# Training_dir = '/media/zhineng/Data/M/aligned_imgs/'
	Training_dir ='/media/nirheaven/nirheaven_ext4/M/aligned_imgs/'

	Save_Model = '/media/zhineng/Data/M/code/code/Models/cnn_cnn_'

	_batch_size = 20
	_seq_num = 32
	_classnum=1400
	learning_rate = 0.00001

	_concat_channel = False
	

if Model == 'multi_channel':
	Print_Labels = False
	Load_Para = True

	Training_dir ='/media/nirheaven/nirheaven_ext4/M/aligned_imgs/'
	Save_Model = '/media/nirheaven/nirheaven_ext4/M/code/code/MC_Models/mc_'
	_batch_size = 60
	_seq_num = 5
	_classnum=1400
	learning_rate = 0.00001

	_concat_channel = True



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



# load net structure
if Model == 'cnn_cnn':
	net = getattr(cnn_cnn,'cnn_cnn_')(batch_size=_batch_size,seq_num=_seq_num,classnum=_classnum,feature=False)
	net_dict = net.state_dict()

	if Load_Pre_Trained:
		load_dict = torch.load(os.getcwd()+'/sphere20a_20171020.pth')
		pretrained_dict = {k: v for k, v in load_dict.items() if k in net_dict}
		net_dict.update(pretrained_dict)
		net.load_state_dict(net_dict)
		# print(net.state_dict())
		# print(net.parameters()) 

if Model == 'multi_channel':
	net = getattr(multi_channel,'mc_')(batch_size=_batch_size,seq_num=_seq_num,classnum=_classnum,feature=False)

	if Load_Para:
		net.load_state_dict(torch.load(Save_Model+'5000.pth'))
		print('successfully load 5000_epoch parameters')
net.to(device)



# load Data

# transform dictionary ('ToTensor' is a must!)
transform = transforms.Compose(
	[transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  

trainset = DatasetFolder(Training_dir,transform=transform,extensions='.jpg',seq_num=_seq_num,concat_channel=_concat_channel)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=_batch_size,
										  shuffle=True, num_workers=1)


# loss function & optimizer
criterion = nn.CrossEntropyLoss()
if Model == 'cnn_cnn':
	#partly fix parameters
	if Fix_preload_para:
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
		optimizer = optim.SGD(para_optim,lr=learning_rate,momentum=0.9)
	else:
		optimizer = optim.SGD(parameters(),lr=learning_rate,momentum=0.9)
if Model == 'multi_channel':
	optimizer = optim.SGD(net.parameters(),lr=learning_rate,momentum=0.9)

# tensorboard visulisation
writer = SummaryWriter(Model)

#training iteration
for epoch in range(5000,15000):  # loop over the dataset multiple times

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

			accuracy = correct/total
			print('Accuray: %d %%'%(100*accuracy))
			writer.add_scalar('train/loss',running_loss / 5, epoch*math.floor(filenum/_batch_size)+ (i+1))
			writer.add_scalar('train/accuracy',accuracy,epoch*math.floor(filenum/_batch_size)+(i+1))
			running_loss = 0.0
			correct = 0
			total = 0
			if Print_Labels:
				print('predicts')
				print(predicted)
				print('labels')
				print(labels)

	if epoch%200 == 199 or epoch == 1:
		os.system('touch '+Save_Model+ str(epoch+1) +'.pth')
		torch.save(net.state_dict(),Save_Model+ str(epoch+1) +'.pth')


# tensorboard visulisation
writer.close()


print('Finished Training')




















