import torch
from torch import nn
import torch.nn as nn
from DA_block import *
from Inception_block import *
from DSConv import DSConv
from thop import profile

from torchsummary import summary


class DAMSLNet(nn.Module):
    def __init__(self,nclasses):
        super(DAMSLNet, self).__init__()
        self.Conv1 = DSConv(3, 64, kernel_size=7, stride=2)
        self.relu = nn.ReLU(inplace=True)
        self.Conv2 = DSConv(64, 128, kernel_size=5, stride=2)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.icpA = InceptionResNetA(128)
        self.redA = ReductionA(128, 768, 192, 224, 256, 384)
        self.icpB = InceptionResNetB(768)
        self.redB = ReductionB(768, 1280)
        self.decoder = DAHead(1280, 1280, nn.BatchNorm2d)
        self.avgpool = nn.AvgPool2d(kernel_size=4)
        self.dropout = nn.Dropout(p=0.8)
        self.linear = nn.Linear(1280, nclasses)
        
    def forward(self, x):
        # Encoder
        out = self.Conv1(x)
        out = self.relu(out)
        out = self.Conv2(out)
        out = self.relu(out)
        # MaxPool
        out = self.maxpool(out)
        # InceptionResNetA
        out = self.icpA(out)
        # ReductionA
        out = self.redA(out)
        # InceptionResNetB
        out = self.icpB(out)
        # ReductionB
        out = self.redB(out)
        # Decoder
        out = self.decoder(out)
        # Global Average Pooling
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        #Dropout
        out = self.dropout(out)
        #Linear(Softmax)
        out = self.linear(out)
        return out

input = torch.randn(1, 3, 256, 256)
flops, params = profile(DAMSLNet(nclasses=8), inputs=(input, ))
 
print("FLOPs=", str(flops/1e9) +'{}'.format("G"))
print("params=", str(params/1e6)+'{}'.format("M"))

# if __name__ == "__main__":
    
#     model = AMSLNet(nclasses=12).cuda()
#     summary(model, input_size=(3, 256, 256), batch_size=1)  # 用于输出网络的构成 参数 和所占内存



