import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, 1) # 128*128*1 -> 128*128*32
        self.conv1_bn = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1) # 128*128*32 -> 64*64*64
        self.conv2_bn = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1) # 64*64*64 -> 32*32*128
        self.conv3_bn = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1) # 32*32*128 -> 16*16*256
        self.conv4_bn = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(256, 512, 3, 2, 1) # 16*16*256 -> 8*8*512
        self.conv5_bn = nn.BatchNorm2d(512)

        self.conv6 = nn.Conv2d(512, 1024, 3, 2, 1) # 8*8*512 -> 4*4*1024
        self.conv6_bn = nn.BatchNorm2d(1024)

        self.conv7 = nn.Conv2d(1024, 2048, 3, 2, 1) # 4*4*1024 -> 2*2*2048
        self.conv7_bn = nn.BatchNorm2d(2048)

        self.convT7 = nn.ConvTranspose2d(2048, 1024, 3, 2, 1, 1) # 2*2*2048 -> 4*4*1024
        self.convT7_bn = nn.BatchNorm2d(1024)
        self.convT7_drop = nn.Dropout2d(0.10)

        self.convT6 = nn.ConvTranspose2d(1024, 512, 3, 2, 1, 1) # 4*4*1024 -> 8*8*512
        self.convT6_bn = nn.BatchNorm2d(512)
        self.convT6_drop = nn.Dropout2d(0.09)

        self.convT5 = nn.ConvTranspose2d(512, 256, 3, 2, 1, 1) # 8*8*512 -> 16*16*256
        self.convT5_bn = nn.BatchNorm2d(256)
        self.convT5_drop = nn.Dropout2d(0.08)

        self.convT4 = nn.ConvTranspose2d(256, 128, 3, 2, 1, 1) # 16*16*256 -> 32*32*128
        self.convT4_bn = nn.BatchNorm2d(128)
        self.convT4_drop = nn.Dropout2d(0.07)

        self.convT3 = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1) # 32*32*128 -> 64*64*64
        self.convT3_bn = nn.BatchNorm2d(64)
        self.convT3_drop = nn.Dropout2d(0.06)

        self.convT2 = nn.ConvTranspose2d(64, 32, 3, 2, 1, 1) # 64*64*64 -> 128*128*32
        self.convT2_bn = nn.BatchNorm2d(32)
        self.convT2_drop = nn.Dropout2d(0.05)

        self.convT1 = nn.ConvTranspose2d(32, 1, 3, 1, 1) # 128*128*32 -> 128*128*1
        self.convT1_bn = nn.BatchNorm2d(1)

    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = F.relu(self.conv6_bn(self.conv6(x)))
        x = torch.tanh(self.conv7_bn(self.conv7(x)))

        embeddings = x.view(x.size(0), -1)

        x = self.convT7_drop(F.relu(self.convT7_bn(self.convT7(x))))
        x = self.convT6_drop(F.relu(self.convT6_bn(self.convT6(x))))
        x = self.convT5_drop(F.relu(self.convT5_bn(self.convT5(x))))
        x = self.convT4_drop(F.relu(self.convT4_bn(self.convT4(x))))
        x = self.convT3_drop(F.relu(self.convT3_bn(self.convT3(x))))
        x = self.convT2_drop(F.relu(self.convT2_bn(self.convT2(x))))
        x = torch.tanh(self.convT1_bn(self.convT1(x)))

        return embeddings, x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # Transition 0
        self.transition0 = nn.ConvTranspose2d(2048, 1024, 3, 2, 1, 1)
        self.transition0_bn = nn.BatchNorm2d(1024)
        self.transition0_drop = nn.Dropout2d(0.6)

        # Dense Block 1
        self.convT1 = nn.ConvTranspose2d(1024, 1024, 3, 1, 1)
        self.convT1_bn = nn.BatchNorm2d(1024)

        self.convT2 = nn.ConvTranspose2d(1024+1024, 1024, 3, 1, 1)
        self.convT2_bn = nn.BatchNorm2d(1024)

        self.convT3 = nn.ConvTranspose2d(1024+1024+1024, 1024, 3, 1, 1)
        self.convT3_bn = nn.BatchNorm2d(1024)

        # Transition 1
        self.transition1 = nn.ConvTranspose2d(1024, 512, 3, 2, 1, 1)
        self.transition1_bn = nn.BatchNorm2d(512)
        self.transition1_drop = nn.Dropout2d(0.5)

        # Dense Block 2
        self.convT4 = nn.ConvTranspose2d(512, 512, 3, 1, 1)
        self.convT4_bn = nn.BatchNorm2d(512)

        self.convT5 = nn.ConvTranspose2d(512+512, 512, 3, 1, 1)
        self.convT5_bn = nn.BatchNorm2d(512)

        self.convT6 = nn.ConvTranspose2d(512+512+512, 512, 3, 1, 1)
        self.convT6_bn = nn.BatchNorm2d(512)

        # Transition 2
        self.transition2 = nn.ConvTranspose2d(512, 256, 3, 2, 1, 1)
        self.transition2_bn = nn.BatchNorm2d(256)
        self.transition2_drop = nn.Dropout2d(0.4)

        # Dense Block 3
        self.convT7 = nn.ConvTranspose2d(256, 256, 3, 1, 1)
        self.convT7_bn = nn.BatchNorm2d(256)

        self.convT8 = nn.ConvTranspose2d(256+256, 256, 3, 1, 1)
        self.convT8_bn = nn.BatchNorm2d(256)

        self.convT9 = nn.ConvTranspose2d(256+256+256, 256, 3, 1, 1)
        self.convT9_bn = nn.BatchNorm2d(256)

        # Transition 3
        self.transition3 = nn.ConvTranspose2d(256, 128, 3, 2, 1, 1)
        self.transition3_bn = nn.BatchNorm2d(128)
        self.transition3_drop = nn.Dropout2d(0.3)

        # Dense Block 4
        self.convT10 = nn.ConvTranspose2d(128, 128, 3, 1, 1)
        self.convT10_bn = nn.BatchNorm2d(128)

        self.convT11 = nn.ConvTranspose2d(128+128, 128, 3, 1, 1)
        self.convT11_bn = nn.BatchNorm2d(128)

        self.convT12 = nn.ConvTranspose2d(128+128+128, 128, 3, 1, 1)
        self.convT12_bn = nn.BatchNorm2d(128)

        # Transition 4
        self.transition4 = nn.ConvTranspose2d(128, 64, 3, 2, 1, 1)
        self.transition4_bn = nn.BatchNorm2d(64)
        self.transition4_drop = nn.Dropout2d(0.2)

        # Dense Block 5
        self.convT13 = nn.ConvTranspose2d(64, 64, 3, 1, 1)
        self.convT13_bn = nn.BatchNorm2d(64)

        self.convT14 = nn.ConvTranspose2d(64+64, 64, 3, 1, 1)
        self.convT14_bn = nn.BatchNorm2d(64)

        self.convT15 = nn.ConvTranspose2d(64+64+64, 64, 3, 1, 1)
        self.convT15_bn = nn.BatchNorm2d(64)

        # Transition 5
        self.transition5 = nn.ConvTranspose2d(64, 32, 3, 2, 1, 1)
        self.transition5_bn = nn.BatchNorm2d(32)
        self.transition5_drop = nn.Dropout2d(0.1)

        # Dense Block 6
        self.convT16 = nn.ConvTranspose2d(32, 32, 3, 1, 1)
        self.convT16_bn = nn.BatchNorm2d(32)

        self.convT17 = nn.ConvTranspose2d(32+32, 32, 3, 1, 1)
        self.convT17_bn = nn.BatchNorm2d(32)

        self.convT18 = nn.ConvTranspose2d(32+32+32, 32, 3, 1, 1)
        self.convT18_bn = nn.BatchNorm2d(32)

        self.transition6 = nn.ConvTranspose2d(32, 3, 1)
        self.transition6_bn = nn.BatchNorm2d(3)

    def forward(self, x):
        x = x.view(x.size(0), 2048, 2, 2)

        # Transition 0
        x = self.transition0_drop(F.relu(self.transition0_bn(self.transition0(x))))

        # Dense Block 1
        x1 = F.relu(self.convT1_bn(self.convT1(x)))
        x1_dense = torch.cat((x, x1), 1)
        x2 = F.relu(self.convT2_bn(self.convT2(x1_dense)))
        x2_dense = torch.cat((x, x1, x2), 1)
        x3 = F.relu(self.convT3_bn(self.convT3(x2_dense)))

        # Transition 1
        x = self.transition1_drop(F.relu(self.transition1_bn(self.transition1(x3))))

        # Dense Block 2
        x1 = F.relu(self.convT4_bn(self.convT4(x)))
        x1_dense = torch.cat((x, x1), 1)
        x2 = F.relu(self.convT5_bn(self.convT5(x1_dense)))
        x2_dense = torch.cat((x, x1, x2), 1)
        x3 = F.relu(self.convT6_bn(self.convT6(x2_dense)))

        # Transition 2
        x = self.transition2_drop(F.relu(self.transition2_bn(self.transition2(x3))))

        # Dense Block 3
        x1 = F.relu(self.convT7_bn(self.convT7(x)))
        x1_dense = torch.cat((x, x1), 1)
        x2 = F.relu(self.convT8_bn(self.convT8(x1_dense)))
        x2_dense = torch.cat((x, x1, x2), 1)
        x3 = F.relu(self.convT9_bn(self.convT9(x2_dense)))

        # Transition 3
        x = self.transition3_drop(F.relu(self.transition3_bn(self.transition3(x3))))

        # Dense Block 4
        x1 = F.relu(self.convT10_bn(self.convT10(x)))
        x1_dense = torch.cat((x, x1), 1)
        x2 = F.relu(self.convT11_bn(self.convT11(x1_dense)))
        x2_dense = torch.cat((x, x1, x2), 1)
        x3 = F.relu(self.convT12_bn(self.convT12(x2_dense)))

        # Transition 4
        x = self.transition4_drop(F.relu(self.transition4_bn(self.transition4(x3))))

        # Dense Block 5
        x1 = F.relu(self.convT13_bn(self.convT13(x)))
        x1_dense = torch.cat((x, x1), 1)
        x2 = F.relu(self.convT14_bn(self.convT14(x1_dense)))
        x2_dense = torch.cat((x, x1, x2), 1)
        x3 = F.relu(self.convT15_bn(self.convT15(x2_dense)))

        # Transition 5
        x = self.transition5_drop(F.relu(self.transition5_bn(self.transition5(x3))))

        # Dense Block 6
        x1 = F.relu(self.convT16_bn(self.convT16(x)))
        x1_dense = torch.cat((x, x1), 1)
        x2 = F.relu(self.convT17_bn(self.convT17(x1_dense)))
        x2_dense = torch.cat((x, x1, x2), 1)
        x3 = F.relu(self.convT18_bn(self.convT18(x2_dense)))

        # Transition 6
        x = torch.tanh(self.transition6_bn(self.transition6(x3)))

        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1) # 128*128*3 -> 128*128*32
        self.conv1_bn = nn.BatchNorm2d(32)
        self.conv1_drop = nn.Dropout2d(0.5)

        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1) # 128*128*32 -> 64*64*64
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv2_drop = nn.Dropout2d(0.5)

        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1) # 64*64*64 -> 32*32*128
        self.conv3_bn = nn.BatchNorm2d(128)
        self.conv3_drop = nn.Dropout2d(0.5)

        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1) # 32*32*128 -> 16*16*256
        self.conv4_bn = nn.BatchNorm2d(256)
        self.conv4_drop = nn.Dropout2d(0.5)

        self.conv5 = nn.Conv2d(256, 512, 3, 2, 1) # 16*16*256 -> 8*8*512
        self.conv5_bn = nn.BatchNorm2d(512)
        self.conv5_drop = nn.Dropout2d(0.5)

        self.conv6 = nn.Conv2d(512, 1024, 3, 2, 1) # 8*8*512 -> 4*4*1024
        self.conv6_bn = nn.BatchNorm2d(1024)
        self.conv6_drop = nn.Dropout2d(0.5)

        self.conv7 = nn.Conv2d(1024, 2048, 3, 2, 1) # 4*4*1024 -> 2*2*2048
        self.conv7_bn = nn.BatchNorm2d(2048)
        self.conv7_drop = nn.Dropout2d(0.5)

        self.conv8 = nn.Conv2d(4096, 1, 3, 2, 1)
        self.conv8_bn = nn.BatchNorm2d(1)
        

    def forward(self, x, embeddings):
        x = self.conv1_drop(F.relu(self.conv1_bn(self.conv1(x))))
        x = self.conv2_drop(F.relu(self.conv2_bn(self.conv2(x))))
        x = self.conv3_drop(F.relu(self.conv3_bn(self.conv3(x))))
        x = self.conv4_drop(F.relu(self.conv4_bn(self.conv4(x))))
        x = self.conv5_drop(F.relu(self.conv5_bn(self.conv5(x))))
        x = self.conv6_drop(F.relu(self.conv6_bn(self.conv6(x))))
        x = self.conv7_drop(F.relu(self.conv7_bn(self.conv7(x))))

        emb_shape = embeddings.size()
        emb = embeddings.view(emb_shape[0], -1, 2, 2)
        x = torch.cat([x, emb], 1)

        x = torch.tanh(self.conv8_bn(self.conv8(x)))

        x = x.view(x.size(0))

        return x

class D(nn.Module):
    def __init__(self):
        super(D, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, 1, 1) # 128*128*1 -> 128*128*32
        self.conv1_bn = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1) # 128*128*32 -> 64*64*64
        self.conv2_bn = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, 2, 1) # 64*64*64 -> 32*32*128
        self.conv3_bn = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, 3, 2, 1) # 32*32*128 -> 16*16*256
        self.conv4_bn = nn.BatchNorm2d(256)

        self.conv5 = nn.Conv2d(256, 512, 3, 2, 1) # 16*16*256 -> 8*8*512
        self.conv5_bn = nn.BatchNorm2d(512)

        self.conv6 = nn.Conv2d(512, 1024, 3, 2, 1) # 8*8*512 -> 4*4*1024
        self.conv6_bn = nn.BatchNorm2d(1024)

        self.conv7 = nn.Conv2d(1024, 2048, 3, 2, 1) # 4*4*1024 -> 2*2*2048
        self.conv7_bn = nn.BatchNorm2d(2048)

        self.li1 = nn.Linear(8192, 1024)

        self.li2 = nn.Linear(1024, 512)

        self.li3 = nn.Linear(512, 256)

        self.li4 = nn.Linear(256, 128)

        self.li5 = nn.Linear(128, 64)

        self.li6 = nn.Linear(64, 32)

        self.li7 = nn.Linear(32, 16)

        self.li8 = nn.Linear(16, 8)
    
    def forward(self, x):
        x = F.relu(self.conv1_bn(self.conv1(x)))
        x = F.relu(self.conv2_bn(self.conv2(x)))
        x = F.relu(self.conv3_bn(self.conv3(x)))
        x = F.relu(self.conv4_bn(self.conv4(x)))
        x = F.relu(self.conv5_bn(self.conv5(x)))
        x = F.relu(self.conv6_bn(self.conv6(x)))
        x = F.relu(self.conv7_bn(self.conv7(x)))

        x = x.view(x.size(0), -1)

        x = F.relu(self.li1(x))
        x = F.relu(self.li2(x))
        x = F.relu(self.li3(x))
        x = F.relu(self.li4(x))
        x = F.relu(self.li5(x))
        x = F.relu(self.li6(x))
        x = F.relu(self.li7(x))
        x = torch.sigmoid(self.li8(x))

        return x