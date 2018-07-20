import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import math
from custom_classes import Conv2d_SAME, MaxPool2d_SAME

class mc_(nn.Module):
	def __init__(self,batch_size=10,width=96,height=112,frame_num=5,classnum=1400,feature=True):
		super(mc_,self).__init__()
		self.batch_size = batch_size
		self.feature = feature
		self.frame_num = frame_num
		self.classnum = classnum
		self.width = width
		self.height = height

		#input = B*(3*framenum)*112*96

		self.sg_conv_1 = Conv2d_SAME(self.frame_num*3,192,5,2)
		self.sg_bn_1 = nn.BatchNorm2d(192,eps=1e-3,momentum=0.9)
		self.sg_relu_1 = nn.ReLU(True)
		self.sg_pool_1  = MaxPool2d_SAME(3,2)

		self.sg_conv_2 = Conv2d_SAME(192, 96, 1, 1)
		self.sg_bn_2   = nn.BatchNorm2d(96, eps=1e-3, momentum=0.9)
		self.sg_relu_2 = nn.ReLU(True)

		self.sg_conv_3 = Conv2d_SAME(96, 96, 3, 1)
		self.sg_bn_3   = nn.BatchNorm2d(96, eps=1e-3, momentum=0.9)
		self.sg_relu_3 = nn.ReLU(True)
		
		self.sg_conv_4 = Conv2d_SAME(96, 96, 1, 1)
		self.sg_bn_4   = nn.BatchNorm2d(96, eps=1e-3, momentum=0.9)
		self.sg_relu_4 = nn.ReLU(True)
		
		self.sg_conv_5 = Conv2d_SAME(96, 96, 3, 1)
		self.sg_bn_5   = nn.BatchNorm2d(96, eps=1e-3, momentum=0.9)
		self.sg_relu_5 = nn.ReLU(True)
		self.sg_pool_5 = MaxPool2d_SAME(3, 2)

		self.sg_conv_6 = Conv2d_SAME(96, 192, 1, 1)
		self.sg_bn_6   = nn.BatchNorm2d(192, eps=1e-3, momentum=0.9)
		self.sg_relu_6 = nn.ReLU(True)

		self.sg_conv_7 = Conv2d_SAME(192, 192, 3, 1)
		self.sg_bn_7   = nn.BatchNorm2d(192, eps=1e-3, momentum=0.9)
		self.sg_relu_7 = nn.ReLU(True)
			
		self.sg_conv_8 = Conv2d_SAME(192, 192, 1, 1)
		self.sg_bn_8   = nn.BatchNorm2d(192, eps=1e-3, momentum=0.9)
		self.sg_relu_8 = nn.ReLU(True)

		self.sg_conv_9 = Conv2d_SAME(192, 192, 3, 1)
		self.sg_bn_9   = nn.BatchNorm2d(192, eps=1e-3, momentum=0.9)
		self.sg_relu_9 = nn.ReLU(True)

		self.sg_conv_10 = Conv2d_SAME(192, 192, 1, 1)
		self.sg_bn_10   = nn.BatchNorm2d(192, eps=1e-3, momentum=0.9)
		self.sg_relu_10 = nn.ReLU(True)

		self.sg_conv_11 = Conv2d_SAME(192, 192, 1, 1)
		self.sg_bn_11   = nn.BatchNorm2d(192, eps=1e-3, momentum=0.9)
		self.sg_relu_11 = nn.ReLU(True)
		self.sg_pool_11 = MaxPool2d_SAME(3, 2)

		self.sg_fc_12   = nn.Linear(192*(self.width//16)*(self.height//16),4096)
		self.sg_bn_12   = nn.BatchNorm1d(4096, eps=1e-3, momentum=0.9)
		self.sg_relu_12 = nn.ReLU(True)
		self.sg_dropout = nn.Dropout()

		self.sg_fc_13   = nn.Linear(4096, 256)
		self.sg_bn_13   = nn.BatchNorm1d(256,eps=1e-3, momentum=0.9)
		self.sg_relu_13 = nn.ReLU(True)

		self.sg_fc      = nn.Linear(256, self.classnum)

	def forward(self, x):
		#input: [b*img_num*3*112*96]
		x = x.view(self.batch_size,self.frame_num*3,112,96) #batch_size * 1 channel * Img_Num * 512

		x = self.sg_pool_1(self.sg_relu_1(self.sg_bn_1(self.sg_conv_1(x))))
		x = self.sg_relu_2(self.sg_bn_2(self.sg_conv_2(x)))
		x = self.sg_relu_3(self.sg_bn_3(self.sg_conv_3(x)))
		x = self.sg_relu_4(self.sg_bn_4(self.sg_conv_4(x)))
		x = self.sg_pool_5(self.sg_relu_5(self.sg_bn_5(self.sg_conv_5(x))))
		x = self.sg_relu_6(self.sg_bn_6(self.sg_conv_6(x)))
		x = self.sg_relu_7(self.sg_bn_7(self.sg_conv_7(x)))
		x = self.sg_relu_8(self.sg_bn_8(self.sg_conv_8(x)))
		x = self.sg_relu_9(self.sg_bn_9(self.sg_conv_9(x)))
		x = self.sg_relu_10(self.sg_bn_10(self.sg_conv_10(x)))
		x = self.sg_pool_11(self.sg_relu_11(self.sg_bn_11(self.sg_conv_11(x))))

		x = x.view(x.size(0),-1)
		x = self.sg_fc_12(x)
		x = self.sg_dropout(self.sg_relu_12(self.sg_bn_12(x)))

		x = self.sg_fc_13(x)
		if self.feature: return x

		x = self.sg_relu_13(self.sg_bn_13(x))
		x = self.sg_fc(x)
		return x

























































