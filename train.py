import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from torchvision import transforms, datasets

# training params
lr = 1e-3
batch_size = 4
num_epoch = 100

data_dir = './data/datasets'
checkpoint_dir = './checkpoints' # train된 네트워크가 저장될 checkpoint 디렉토리
log_dir = './logs'               # tensor board 로그 파일 저장

# train이 cpu / gpu 머신에서 동작할 지 결정해주는 device flag
device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

import torch
import torch.nn as nn

# unet
class UNet(nn.Module):
    # unet을 정의하는 데 필요한 layer 선언
    def __init__(self):
        super(UNet, self).__init__()
    
        # convolution, batch normalization, ReLU Layer
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            # convolution layer 정의
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                                kernel_size=kernel_size, stride=stride, padding=padding, 
                                bias = bias)]
            # batch normalization layer
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            # ReLU Layer
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)

            return cbr
    
        # contracting path
        self.encoder1_1 = CBR2d(in_channels=1, out_channels=64)  
        self.encoder1_2 = CBR2d(in_channels=64, out_channels=64)

        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.encoder2_1 = CBR2d(in_channels=64, out_channels=128)
        self.encoder2_2 = CBR2d(in_channels=128, out_channels=128)

        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        self.encoder3_1 = CBR2d(in_channels=128, out_channels=256)
        self.encoder3_2 = CBR2d(in_channels=256, out_channels=256)

        self.pool3 = nn.MaxPool2d(kernel_size=2)
        
        self.encoder4_1 = CBR2d(in_channels=256, out_channels=512)
        self.encoder4_2 = CBR2d(in_channels=512, out_channels=512)

        self.pool4 = nn.MaxPool2d(kernel_size=2)
        
        self.encoder5_1 = CBR2d(in_channels=512, out_channels=1024)

        # expansive path
        self.decoder5_1 = CBR2d(in_channels=1024, out_channels=512)

        self.unpool4 = nn.ConvTranspose2d(in_channels=512, out_channels=512, kernel_size=2, stride=2, padding=0, bias=True)

        self.decoder4_2 = CBR2d(in_channels= 2 * 512, out_channels=512)
        self.decoder4_1 = CBR2d(in_channels=512, out_channels=256)
        
        self.unpool3 = nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0, bias=True)

        self.decoder3_2 = CBR2d(in_channels=2 * 256, out_channels=256)
        self.decoder3_1 = CBR2d(in_channels=256, out_channels=128)
        
        self.unpool2 = nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=2, stride=2, padding=0, bias=True)
        
        self.decoder2_2 = CBR2d(in_channels=2 * 128, out_channels=128)
        self.decoder2_1 = CBR2d(in_channels=128, out_channels=64)
        
        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0, bias=True)

        self.decoder1_2 = CBR2d(in_channels=2 * 64, out_channels=64)
        self.decoder1_1 = CBR2d(in_channels=64, out_channels=64)
        
        self.fc = nn.Conv2d(in_channels=64, out_channels=2, kernel_size=1, stride=1, padding=0, bias=True)

    # unet layer 연결하기
    # forward func -> init 에서 생성한 layer를 연결하는 코드 작성, x: input img
    def forward(self, x):
        # encoder part
        encoder1_1 = self.encoder1_1(x)
        encoder1_2 = self.encoder1_2(encoder1_1)
        pool1 = self.pool1(encoder1_2)

        encoder2_1 = self.encoder2_1(pool1)
        encoder2_2 = self.encoder2_2(encoder2_1)
        pool2 = self.pool2(encoder2_2)
        
        encoder3_1 = self.encoder3_1(pool2)
        encoder3_2 = self.encoder3_2(encoder3_1)
        pool3 = self.pool3(encoder3_2)

        encoder4_1 = self.encoder4_1(pool3)
        encoder4_2 = self.encoder4_2(encoder4_1)
        pool4 = self.pool4(encoder4_2)

        encoder5_1 = self.encoder5_1(pool4)

        # decoder part        
        decoder5_1 = self.decoder5_1(encoder5_1)

        unpool4 = self.unpool4(decoder5_1)
        concat4 = torch.cat((unpool4, encoder4_2), dim=1)
        decoder4_2 = self.decoder4_2(concat4)
        decoder4_1 = self.decoder4_1(decoder4_2)

        unpool3 = self.unpool3(decoder4_1)
        concat3 = torch.cat((unpool3, encoder3_2), dim=1)
        decoder3_2 = self.decoder3_2(concat3)
        decoder3_1 = self.decoder3_1(decoder3_2)

        unpool2 = self.unpool2(decoder3_1)
        concat2 = torch.cat((unpool2, encoder2_2), dim=1)
        decoder2_2 = self.decoder2_2(concat2)
        decoder2_1 = self.decoder2_1(decoder2_2)

        unpool1 = self.unpool1(decoder2_1)
        concat1 = torch.cat((unpool1, encoder1_2), dim=1)
        decoder1_2 = self.decoder1_2(concat1)
        decoder1_1 = self.decoder1_1(decoder1_2)

        x = self.fc(decoder1_1)

        return x