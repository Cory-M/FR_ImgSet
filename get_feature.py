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

from input_pipeline import DatasetFolder

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

Model = 'multi_channel'  #choose from {'cnn_cnn','multi_channel'}
load_epoch_num =10200

#参数设定
if Model == 'cnn_cnn':

	Test_dir ='/media/nirheaven/nirheaven_ext4/M/aligned_imgs_test/'
	Load_Model = '/media/zhineng/Data/M/code/code/Models/cnn_cnn_'
	save_txt = ''

	# net parameters
	_batch_size = 20
	_seq_num = 32
	_classnum=1400
	learning_rate = 0.00001
	_concat_channel = False
	

if Model == 'multi_channel':

	Test_dir ='/media/nirheaven/nirheaven_ext4/M/aligned_imgs_test/'
	Load_Model = '/media/nirheaven/nirheaven_ext4/M/code/code/MC_Models/mc_'
	save_txt = '/media/nirheaven/nirheaven_ext4/M/code/code/feature/mc_epoch'

	#net parameters
	_batch_size = 15
	_seq_num = 5
	_classnum=1400
	learning_rate = 0.00001
	_concat_channel = True


#get file number
filenum = 0
for speaker in os.listdir(Test_dir):
	for video in os.listdir(Test_dir+'/'+speaker):
		filenum+=1
epoch_filenum = math.floor(filenum/_batch_size)*_batch_size
print('File Number is %d, Epoch File Number is %d'%(filenum,epoch_filenum))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Assume that we are on a CUDA machine, then this should print a CUDA device:
print(device)

# load net structure
if Model == 'cnn_cnn':
	net = getattr(cnn_cnn,'cnn_cnn_')(batch_size=_batch_size,seq_num=_seq_num,classnum=_classnum,feature=True)

elif Model == 'multi_channel':
	net = getattr(multi_channel,'mc_')(batch_size=_batch_size,seq_num=_seq_num,classnum=_classnum,feature=True)

else:
	print('load model ERROR: Model does not exist')

net.load_state_dict(torch.load(Load_Model+str(load_epoch_num)+'.pth'))
print('successfully load epoch_%d parameters'%(load_epoch_num))

net.to(device)

transform = transforms.Compose(
	[transforms.ToTensor(),
	transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  

testset = DatasetFolder(Test_dir,transform=transform,extensions='.jpg',seq_num=_seq_num,concat_channel=_concat_channel)

testloader = torch.utils.data.DataLoader(testset, batch_size=_batch_size,
										  shuffle=False, num_workers=1)


for epoch in range(10):
	with open(save_txt+str(load_epoch_num)+'_'+str(epoch)+'.txt','w+') as f:

		for i,data in enumerate(testloader,0):
			if i>(math.floor(filenum/_batch_size)-1): continue

			inputs, labels = data
			inputs = inputs.to(device)

			with torch.no_grad():
				outputs = net(inputs)
				outputs = outputs.to('cpu')

				for i in range(_batch_size):
					f.write(str(labels.numpy()[i])+' '+str(outputs.numpy()[i])+'\n')

















































