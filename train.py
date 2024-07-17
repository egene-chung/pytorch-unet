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
    
## 데이터 로더 구현하기
class Dataset(torch.utils.data.Dataset): # Dataset 크ㄹ래스 상속 받기
    def __init__(self, data_dir, transform=None): # 차후 구현할 것들을 argument로 받기
        self.data_dir = data_dir
        self.transform = transform

        # dataset directory에 저장되어있는 모든 dataset list 얻어와야함.
        # 어떤 식으로 dataset directory에 저장이 되어있는지 확인해야함.
        # input과 label로 prefix되어있는 dataset
        lst_data = os.listdir(self.data_dir)

        lst_label = [f for f in lst_data if f.startswith('label')]
        lst_input = [f for f in lst_data if f.startswith('input')]

        # 리스트 정렬
        lst_label.sort()
        lst_input.sort()

        # 이를 파라미터로 가지고 있기.
        self.lst_label = lst_label
        self.lst_input = lst_input
    
    def __len__(self):
        return len(self.lst_label)
    
    def __getitem__(self, index):
        # Numpy형태로 데이터셋이 저장되어있기 때문에 numpy 노드를 이용해 데이터셋 불러오기
        label = np.load(os.path.join(self.data_dir, self.lst_label[index]))
        input = np.load(os.path.join(self.data_dir, self.lst_label[index]))

        # 저장된 데이터가 0-255 -> 0-1로 normalize하기 위해 아래와 같이! -> 왜??
        label = label/255.0
        input = input/255.0

        # neural network에 들어가는 input은 3개의 axis를 가져야함.
        # channel이 없는 경우 (x,y axis만 있는 경우) -> channel에 해당하는 axis를 임의로 생성해줘야함. 
        # numpy의 newaxis 사용
        if label.ndim == 2:
            label = label[:, :, np.newaxis]

        if input.ndim == 2:
            input = input[:, :, np.newaxis]

        data = {'input': input, 'label': label}

        # 만약 transform function을 data_loader의 arg로 넣어준다면, 
        if self.transform:
            data = self.transform(data)
        return data

# dataset class에 해당하는 object 만들기
dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'))

data = dataset_train.__getitem__(0) # 첫번째 index에 해당하는 dataset 불러오기

input = data['input']
label = data['label']

print(label.shape) # (512, 512, 1)

# transform 구현하기
# 필수적으로 들어가야하는 transform 
# ToTensor() : numpy -> tensor
# data = {'input': input, 'label': label} -> input과 label을 dictionary로 갖는 
# data라는 object를 받아서, tensor로 변경
class ToTensor(object): 
    def __call__(self, data):
        label, input = data['label'], data['input']

        # tensor로 담기 전, pytorch의 기본적인 data variable의 data axis와 다름. 
        # img의 numpy 차원 = (Y, X, CH)
        # pytorch = (CH, Y, X) channel dimension의 위치가 다르기 때문에 2로
        label = label.transpose((2, 0, 1)).astype(np.float32)
        input = input.transpose((2, 0, 1)).astype(np.float32)
        
        # numpy를 tensor로 넘겨주는 함수인 from_numpy 사용
        data = {'label': torch.from_numpy(label), 'input': torch.from_numpy(input)}
        
        return data
    
class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        label, input = data['label'], data['input']
        
        input = (input - self.mean) / self.std
        #label같은 경우에는, 0 또는 1의 class로 정의되어있기 때문에 하면 안됨.
        data = {'label': label, 'input': input}
        
        return data

class RandomFlip(object):
    def __call__(self, data):
        label, input = data['label'], data['input']

        # 무조건 label, input 같이
        if np.random.rand() > 0.5:
            label = np.fliplr(label)
            input = np.fliplr(input)
        
        if np.random.rand() > 0.5:
            label = np.flipud(label)
            input = np.flipud(input)
        
        data = {'label': label, 'input': input}

        return data

# torch vision의 이미 선언된 transforms 에 정의된 여러 transform 함수를 묶어서 선언할 수 있는 compose 함수가 있음.
transform = transforms.Compose([Normalization(mean=0.5, std=0.5), RandomFlip(), ToTensor()])
dataset_train = Dataset(data_dir=os.path.join(data_dir, 'train'), transform=transform)

data = dataset_train.__getitem__(0) # 첫번째 index에 해당하는 dataset 불러오기

input = data['input']
label = data['label']

print(label.shape) # torch.Size([1, 512, 512])
print(label.type()) # torch.FloatTensor
 




