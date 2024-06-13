from torchvision import transforms as transforms
import torch.nn.functional as F
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(1, 32, 3, 2)
        self.conv1_bn = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, 2)
        self.conv2_bn = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, 2)
        self.conv3_bn = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, 3, 2)
        self.conv4_bn = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(256, 512, 3, 2)
        self.conv5_bn = nn.BatchNorm2d(512)

        self.conv6 = nn.Conv2d(512, 1024, 3, 2)
        self.conv6_bn = nn.BatchNorm2d(1024)

        self.convT1 = nn.ConvTranspose2d(1024, 512, 3, 2)
        self.convT1_bn = nn.BatchNorm2d(512)
        self.convT1_drop = nn.Dropout2d(0.09)

        self.convT2 = nn.ConvTranspose2d(512, 256, 3, 2)
        self.convT2_bn = nn.BatchNorm2d(256)
        self.convT2_drop = nn.Dropout2d(0.08)

        self.convT3 = nn.ConvTranspose2d(256, 128, 3, 2)
        self.convT3_bn = nn.BatchNorm2d(128)
        self.convT3_drop = nn.Dropout2d(0.07)

        self.convT4 = nn.ConvTranspose2d(128, 64, 3, 2)
        self.convT4_bn = nn.BatchNorm2d(64)
        self.convT4_drop = nn.Dropout2d(0.06)

        self.convT5 = nn.ConvTranspose2d(64, 32, 3, 2)
        self.convT5_bn = nn.BatchNorm2d(32)
        self.convT5_drop = nn.Dropout2d(0.05)
        
        self.convT6 = nn.ConvTranspose2d(32, 3, 3, 2)
        self.convT6_bn = nn.BatchNorm2d(3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.conv4_bn(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.conv5_bn(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.conv6_bn(x)
        x = self.relu(x)

        x = self.convT1(x)
        x = self.convT1_bn(x)
        x = self.relu(x)
        x = self.convT1_drop(x)
        x = self.convT2(x)
        x = self.convT2_bn(x)
        x = self.relu(x)
        x = self.convT2_drop(x)
        x = self.convT3(x)
        x = self.convT3_bn(x)
        x = self.relu(x)
        x = self.convT3_drop(x)
        x = self.convT4(x)
        x = self.convT4_bn(x)
        x = self.relu(x)
        x = self.convT4_drop(x)
        x = self.convT5(x)
        x = self.convT5_bn(x)
        x = self.relu(x)
        x = self.convT5_drop(x)
        x = self.convT6(x)
        x = self.convT6_bn(x)
        return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(1, 32, 3, 2)
        self.conv1_bn = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, 2)
        self.conv2_bn = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, 2)
        self.conv3_bn = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, 3, 2)
        self.conv4_bn = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(256, 512, 3, 2)
        self.conv5_bn = nn.BatchNorm2d(512)

        self.conv6 = nn.Conv2d(512, 1024, 3, 2)
        self.conv6_bn = nn.BatchNorm2d(1024)

        self.convT1 = nn.ConvTranspose2d(1024, 512, 3, 2)
        self.convT1_bn = nn.BatchNorm2d(512)
        self.convT1_drop = nn.Dropout2d(0.09)

        self.convT2 = nn.ConvTranspose2d(512, 256, 3, 2)
        self.convT2_bn = nn.BatchNorm2d(256)
        self.convT2_drop = nn.Dropout2d(0.08)

        self.convT3 = nn.ConvTranspose2d(256, 128, 3, 2)
        self.convT3_bn = nn.BatchNorm2d(128)
        self.convT3_drop = nn.Dropout2d(0.07)

        self.convT4 = nn.ConvTranspose2d(128, 64, 3, 2)
        self.convT4_bn = nn.BatchNorm2d(64)
        self.convT4_drop = nn.Dropout2d(0.06)

        self.convT5 = nn.ConvTranspose2d(64, 32, 3, 2)
        self.convT5_bn = nn.BatchNorm2d(32)
        self.convT5_drop = nn.Dropout2d(0.05)
        
        self.convT6 = nn.ConvTranspose2d(32, 3, 3, 2)
        self.convT6_bn = nn.BatchNorm2d(3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.conv4_bn(x)
        x = self.relu(x)
        x = self.conv5(x)
        x = self.conv5_bn(x)
        x = self.relu(x)
        x = self.conv6(x)
        x = self.conv6_bn(x)
        x = self.relu(x)

        x = self.convT1(x)
        x = self.convT1_bn(x)
        x = self.relu(x)
        x = self.convT1_drop(x)
        x = self.convT2(x)
        x = self.convT2_bn(x)
        x = self.relu(x)
        x = self.convT2_drop(x)
        x = self.convT3(x)
        x = self.convT3_bn(x)
        x = self.relu(x)
        x = self.convT3_drop(x)
        x = self.convT4(x)
        x = self.convT4_bn(x)
        x = self.relu(x)
        x = self.convT4_drop(x)
        x = self.convT5(x)
        x = self.convT5_bn(x)
        x = self.relu(x)
        x = self.convT5_drop(x)
        x = self.convT6(x)
        x = self.convT6_bn(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3, 32, 3, 2)
        self.conv1_bn = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, 2)
        self.conv2_bn = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, 2)
        self.conv3_bn = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, 3, 2)
        self.conv4_bn = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(256, 512, 3, 2)
        self.conv5_bn = nn.BatchNorm2d(512)

        self.conv6 = nn.Conv2d(512, 1, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = self.relu(x)

        x = self.conv4(x)
        x = self.conv4_bn(x)
        x = self.relu(x)

        x = self.conv5(x)
        x = self.conv5_bn(x)
        x = self.relu(x)

        x = self.conv6(x)
        return x