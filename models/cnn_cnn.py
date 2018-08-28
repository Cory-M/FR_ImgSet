import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import math
from custom_classes import Conv2d_SAME, MaxPool2d_SAME


class cnn_cnn_(nn.Module):
    def __init__(self,feature=True,seq_num=16,final_size_w=512,classnum=1400,batch_size=2):
        super(cnn_cnn_, self).__init__()
        self.classnum = classnum
        self.feature = feature
        self.seq_num = seq_num
        self.final_size_w = final_size_w
        self.batch_size = batch_size
        #input = B*3*112*96
        self.conv1_1 = nn.Conv2d(3,64,3,2,1) #=>B*64*56*48
        self.relu1_1 = nn.PReLU(64)
        self.conv1_2 = nn.Conv2d(64,64,3,1,1)
        self.relu1_2 = nn.PReLU(64)
        self.conv1_3 = nn.Conv2d(64,64,3,1,1)
        self.relu1_3 = nn.PReLU(64)

        self.conv2_1 = nn.Conv2d(64,128,3,2,1) #=>B*128*28*24
        self.relu2_1 = nn.PReLU(128)
        self.conv2_2 = nn.Conv2d(128,128,3,1,1)
        self.relu2_2 = nn.PReLU(128)
        self.conv2_3 = nn.Conv2d(128,128,3,1,1)
        self.relu2_3 = nn.PReLU(128)

        self.conv2_4 = nn.Conv2d(128,128,3,1,1) #=>B*128*28*24
        self.relu2_4 = nn.PReLU(128)
        self.conv2_5 = nn.Conv2d(128,128,3,1,1)
        self.relu2_5 = nn.PReLU(128)


        self.conv3_1 = nn.Conv2d(128,256,3,2,1) #=>B*256*14*12
        self.relu3_1 = nn.PReLU(256)
        self.conv3_2 = nn.Conv2d(256,256,3,1,1)
        self.relu3_2 = nn.PReLU(256)
        self.conv3_3 = nn.Conv2d(256,256,3,1,1)
        self.relu3_3 = nn.PReLU(256)

        self.conv3_4 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_4 = nn.PReLU(256)
        self.conv3_5 = nn.Conv2d(256,256,3,1,1)
        self.relu3_5 = nn.PReLU(256)

        self.conv3_6 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_6 = nn.PReLU(256)
        self.conv3_7 = nn.Conv2d(256,256,3,1,1)
        self.relu3_7 = nn.PReLU(256)

        self.conv3_8 = nn.Conv2d(256,256,3,1,1) #=>B*256*14*12
        self.relu3_8 = nn.PReLU(256)
        self.conv3_9 = nn.Conv2d(256,256,3,1,1)
        self.relu3_9 = nn.PReLU(256)

        self.conv4_1 = nn.Conv2d(256,512,3,2,1) #=>B*512*7*6
        self.relu4_1 = nn.PReLU(512)
        self.conv4_2 = nn.Conv2d(512,512,3,1,1)
        self.relu4_2 = nn.PReLU(512)
        self.conv4_3 = nn.Conv2d(512,512,3,1,1)
        self.relu4_3 = nn.PReLU(512)

        self.fc5 = nn.Linear(512*7*6,512) ##=>B*Img*512
        


        self.sg_conv_1 = Conv2d_SAME(1,48,5,2) #in_channel, out_channel, kernel, stride
        self.sg_bn_1   = nn.BatchNorm2d(48, eps=1e-3, momentum=0.9)
        self.sg_relu_1 = nn.ReLU(True)
        self.sg_pool_1 = MaxPool2d_SAME(3, 2)

        self.sg_conv_2 = Conv2d_SAME(48, 96, 1, 1)
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

        self.sg_fc_12   = nn.Linear(192*(self.final_size_w//16)*(self.seq_num//16),4096)
        self.sg_bn_12   = nn.BatchNorm1d(4096, eps=1e-3, momentum=0.9)
        self.sg_relu_12 = nn.ReLU(True)
        self.sg_dropout = nn.Dropout()

        self.sg_fc_13   = nn.Linear(4096, 256)
        self.sg_bn_13   = nn.BatchNorm1d(256,eps=1e-3, momentum=0.9)
        self.sg_relu_13 = nn.ReLU(True)

        self.sg_fc      = nn.Linear(256, self.classnum)

    def forward(self, x):
        #input: [b*img_num*3*112*96]
        x = x.view(-1,3,112,96) #get individual imgs
        x = self.relu1_1(self.conv1_1(x))
        x = x + self.relu1_3(self.conv1_3(self.relu1_2(self.conv1_2(x))))

        x = self.relu2_1(self.conv2_1(x))
        x = x + self.relu2_3(self.conv2_3(self.relu2_2(self.conv2_2(x))))
        x = x + self.relu2_5(self.conv2_5(self.relu2_4(self.conv2_4(x))))

        x = self.relu3_1(self.conv3_1(x))
        x = x + self.relu3_3(self.conv3_3(self.relu3_2(self.conv3_2(x))))
        x = x + self.relu3_5(self.conv3_5(self.relu3_4(self.conv3_4(x))))
        x = x + self.relu3_7(self.conv3_7(self.relu3_6(self.conv3_6(x))))
        x = x + self.relu3_9(self.conv3_9(self.relu3_8(self.conv3_8(x))))

        x = self.relu4_1(self.conv4_1(x))
        x = x + self.relu4_3(self.conv4_3(self.relu4_2(self.conv4_2(x))))

        x = x.view(x.size(0),-1)
        x = self.fc5(x)

        x = x.view(self.batch_size,1,-1,512) #batch_size * 1 channel * Img_Num * 512

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








