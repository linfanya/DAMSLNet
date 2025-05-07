import torch
from torch import nn
import torch.nn.functional as F

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x)

class Stem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Stem, self).__init__()
        #conv3*3(32 stride2 valid)
        self.conv1 = BasicConv2d(in_channels, 32, kernel_size=3, stride=2)
        #conv3*3(32 valid)
        self.conv2 = BasicConv2d(32, 32, kernel_size=3)
        #conv3*3(64)
        self.conv3 = BasicConv2d(32, 64, kernel_size=3, padding=1)
        #maxpool3*3(stride2 valid) & conv3*3(96 stride2 valid)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv4 = BasicConv2d(64, 96, kernel_size=3, stride=2)
        
        #conv1*1(64) --> conv3*3(96 valid)
        self.conv5_1_1 = BasicConv2d(160, 64, kernel_size=1)
        self.conv5_1_2 = BasicConv2d(64, 96, kernel_size=3)
        #conv1*1(64) --> conv7*1(64) --> conv1*7(64) --> conv3*3(96 valid)
        self.conv5_2_1 = BasicConv2d(160, 64, kernel_size=1)
        self.conv5_2_2 = BasicConv2d(64, 64, kernel_size=(7,1), padding=(3,0))
        self.conv5_2_3 = BasicConv2d(64, 64, kernel_size=(1,7), padding=(0,3))
        self.conv5_2_4 = BasicConv2d(64, 96, kernel_size=3)
        
        #conv3*3(192 valid)
        self.conv6 = BasicConv2d(192, 192, kernel_size=3, stride=2)
        #maxpool3*3(stride2 valid)
        self.maxpool6 = nn.MaxPool2d(kernel_size=3, stride=2)
        
    def forward(self, x):
        y1_1 = self.maxpool4(self.conv3(self.conv2(self.conv1(x))))
        y1_2 = self.conv4(self.conv3(self.conv2(self.conv1(x))))
        y1 = torch.cat([y1_1, y1_2], 1)
        
        y2_1 = self.conv5_1_2(self.conv5_1_1(y1))
        y2_2 = self.conv5_2_4(self.conv5_2_3(self.conv5_2_2(self.conv5_2_1(y1))))
        y2 = torch.cat([y2_1, y2_2], 1)
        
        y3_1 = self.conv6(y2)
        y3_2 = self.maxpool6(y2)
        y3 = torch.cat([y3_1, y3_2], 1)
        
        return y3

class InceptionA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionA, self).__init__()
        #branch1: avgpool --> conv1*1(96)
        self.b1_1 = nn.AvgPool2d(kernel_size=3, padding=1, stride=1)
        self.b1_2 = BasicConv2d(in_channels, 96, kernel_size=1)
        
        #branch2: conv1*1(96)
        self.b2 = BasicConv2d(in_channels, 96, kernel_size=1)
        
        #branch3: conv1*1(64) --> conv3*3(96)
        self.b3_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.b3_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        
        #branch4: conv1*1(64) --> conv3*3(96) --> conv3*3(96)
        self.b4_1 = BasicConv2d(in_channels, 64, kernel_size=1)
        self.b4_2 = BasicConv2d(64, 96, kernel_size=3, padding=1)
        self.b4_3 = BasicConv2d(96, 96, kernel_size=3, padding=1)
        
    def forward(self, x):
        y1 = self.b1_2(self.b1_1(x))
        y2 = self.b2(x)
        y3 = self.b3_2(self.b3_1(x))
        y4 = self.b4_3(self.b4_2(self.b4_1(x)))
        
        outputsA = [y1, y2, y3, y4]
        return torch.cat(outputsA, 1)

class InceptionB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionB, self).__init__()
        #branch1: avgpool --> conv1*1(128)
        self.b1_1 = nn.AvgPool2d(kernel_size=3, padding=1, stride=1)
        self.b1_2 = BasicConv2d(in_channels, 128, kernel_size=1)
        
        #branch2: conv1*1(384)
        self.b2 = BasicConv2d(in_channels, 384, kernel_size=1)
        
        #branch3: conv1*1(192) --> conv1*7(224) --> conv1*7(256)
        self.b3_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.b3_2 = BasicConv2d(192, 224, kernel_size=(7,1), padding=(3,0))
        self.b3_3 = BasicConv2d(224, 256, kernel_size=(1,7), padding=(0,3))
        
        #branch4: conv1*1(192) --> conv1*7(192) --> conv7*1(224) --> conv1*7(224) --> conv7*1(256)
        self.b4_1 = BasicConv2d(in_channels, 192, kernel_size=1, stride=1)
        self.b4_2 = BasicConv2d(192, 192, kernel_size=(1,7), padding=(0,3))
        self.b4_3 = BasicConv2d(192, 224, kernel_size=(7,1), padding=(3,0))
        self.b4_4 = BasicConv2d(224, 224, kernel_size=(1,7), padding=(0,3))
        self.b4_5 = BasicConv2d(224, 256, kernel_size=(7,1), padding=(3,0))
        
    def forward(self, x):
        y1 = self.b1_2(self.b1_1(x))
        y2 = self.b2(x)
        y3 = self.b3_3(self.b3_2(self.b3_1(x)))
        y4 = self.b4_5(self.b4_4(self.b4_3(self.b4_2(self.b4_1(x)))))
        
        outputsB = [y1, y2, y3, y4]
        return torch.cat(outputsB, 1)

class InceptionC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionC, self).__init__()
        #branch1: avgpool --> conv1*1(256)
        self.b1_1 = nn.AvgPool2d(kernel_size=3, padding=1, stride=1)
        self.b1_2 = BasicConv2d(in_channels, 256, kernel_size=1)
        
        #branch2: conv1*1(256)
        self.b2 = BasicConv2d(in_channels, 256, kernel_size=1)
        
        #branch3: conv1*1(384) --> conv1*3(256) & conv3*1(256)
        self.b3_1 = BasicConv2d(in_channels, 384, kernel_size=1)
        self.b3_2_1 = BasicConv2d(384, 256, kernel_size=(1,3), padding=(0,1))
        self.b3_2_2 = BasicConv2d(384, 256, kernel_size=(3,1), padding=(1,0))
        
        #branch4: conv1*1(384) --> conv1*3(448) --> conv3*1(512) --> conv3*1(256) & conv7*1(256)
        self.b4_1 = BasicConv2d(in_channels, 384, kernel_size=1, stride=1)
        self.b4_2 = BasicConv2d(384, 448, kernel_size=(1,3), padding=(0,1))
        self.b4_3 = BasicConv2d(448, 512, kernel_size=(3,1), padding=(1,0))
        self.b4_4_1 = BasicConv2d(512, 256, kernel_size=(3,1), padding=(1,0))
        self.b4_4_2 = BasicConv2d(512, 256, kernel_size=(1,3), padding=(0,1))
        
    def forward(self, x):
        y1 = self.b1_2(self.b1_1(x))
        y2 = self.b2(x)
        y3_1 = self.b3_2_1(self.b3_1(x))
        y3_2 = self.b3_2_2(self.b3_1(x))
        y4_1 = self.b4_4_1(self.b4_3(self.b4_2(self.b4_1(x))))
        y4_2 = self.b4_4_2(self.b4_3(self.b4_2(self.b4_1(x))))
        
        outputsC = [y1, y2, y3_1, y3_2, y4_1, y4_2]
        return torch.cat(outputsC, 1)
    

class SEnet(nn.Module):
    def __init__(self, channel, ratio=16):
        super(SEnet,self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel//ratio, False),
            nn.ReLU(),
            nn.Linear(channel//ratio, channel, False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        b, c, h, w = x.size()
        avg = self.avg_pool(x).view([b, c])
        fc = self.fc(avg).view([b, c, 1, 1])

        return x * fc


class ReductionA(nn.Module):
    def __init__(self, in_channels, out_channels, k, l, m, n):
        super(ReductionA, self).__init__()
        #branch1: maxpool3*3(stride2 valid)
        self.b1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        #branch2: conv3*3(n stride2 valid)
        self.b2 = BasicConv2d(in_channels, n, kernel_size=3, stride=2)
        
        #branch3: conv1*1(k) --> conv3*3(l) --> conv3*3(m stride2 valid)
        self.b3_1 = BasicConv2d(in_channels, k, kernel_size=1)
        self.b3_2 = BasicConv2d(k, l, kernel_size=3, padding=1)
        self.b3_3 = BasicConv2d(l, m, kernel_size=3, stride=2)
        

        
    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2(x)
        y3 = self.b3_3(self.b3_2(self.b3_1(x)))
        
        outputsRedA = [y1, y2, y3]
        out = torch.cat(outputsRedA, 1)
        return out

class ReductionB(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ReductionB, self).__init__()
        #branch1: maxpool3*3(stride2 valid)
        self.b1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        #branch2: conv1*1(192) --> conv3*3(192 stride2 valid)
        self.b2_1 = BasicConv2d(in_channels, 192, kernel_size=1)
        self.b2_2 = BasicConv2d(192, 192, kernel_size=3, stride=2)
        
        #branch3: conv1*1(256) --> conv1*7(256) --> conv7*1(320) --> conv3*3(320 stride2 valid)
        self.b3_1 = BasicConv2d(in_channels, 256, kernel_size=1)
        self.b3_2 = BasicConv2d(256, 256, kernel_size=(1,7), padding=(0,3))
        self.b3_3 = BasicConv2d(256, 320, kernel_size=(7,1), padding=(3,0))
        self.b3_4 = BasicConv2d(320, 320, kernel_size=3, stride=2)

        
    def forward(self, x):
        y1 = self.b1(x)
        y2 = self.b2_2(self.b2_1((x)))
        y3 = self.b3_4(self.b3_3(self.b3_2(self.b3_1(x))))
        
        outputsRedB = [y1, y2, y3]
        out = torch.cat(outputsRedB, 1)
        return out
    

class InceptionResNetA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionResNetA, self).__init__()
        #branch1: conv1*1(32)
        self.b1 = BasicConv2d(in_channels, 32, kernel_size=1)
        
        #branch2: conv1*1(32) --> con3*3(32)
        self.b2_1 = BasicConv2d(in_channels, 32, kernel_size=1)
        self.b2_2 = BasicConv2d(32, 32, kernel_size=3, padding=1)
        
        #branch3: conv1*1(32) --> conv3*3(32) --> conv3*3(32)
        self.b3_1 = BasicConv2d(in_channels, 32, kernel_size=1)
        self.b3_2 = BasicConv2d(32, 32, kernel_size=3, padding=1)
        self.b3_3 = BasicConv2d(32, 32, kernel_size=3, padding=1)
        
        #totalbranch: conv1*1(128)
        self.tb = BasicConv2d(96, 128, kernel_size=1)

        # self.se = SEnet(128)
        
    def forward(self, x):
        b_out1 = self.b1(x)
        b_out2 = self.b2_2(self.b2_1(x))
        b_out3 = self.b3_3(self.b3_2(F.relu(self.b3_1(x))))
        b_out = torch.cat([b_out1, b_out2, b_out3], 1)
        b_out = self.tb(b_out)
        y = b_out + x
        out = F.relu(y)
        # out = self.se(out)
                           
        return out

class InceptionResNetB(nn.Module):
    def __init__(self, in_channels):
        super(InceptionResNetB, self).__init__()
        #branch1: conv1*1(128)
        self.b1 = BasicConv2d(in_channels, 128, kernel_size=1)
        
        #branch2: conv1*1(128) --> con1*7(128) --> conv7*1(128)
        self.b2_1 = BasicConv2d(in_channels, 128, kernel_size=1)
        self.b2_2 = BasicConv2d(128, 128, kernel_size=(1,7), padding=(0,3))
        self.b2_3 = BasicConv2d(128, 128, kernel_size=(7,1), padding=(3,0))
    
        #totalbranch: conv1*1(1024)
        self.tb = BasicConv2d(256, 768, kernel_size=1)
        
    def forward(self, x):
        b_out1 = self.b1(x)
        b_out2 = self.b2_3(self.b2_2(self.b2_1(x)))
        b_out = torch.cat([b_out1, b_out2], 1)
        b_out = self.tb(b_out)
        y = b_out + x
        out = F.relu(y)
                           
        return out
