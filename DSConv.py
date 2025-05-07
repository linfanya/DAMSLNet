from torch import nn

class DSConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride):
        super(DSConv, self).__init__()
 
        # Depthwise Convolution
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,           # input channels
            out_channels=in_ch,          # output channels
            kernel_size=kernel_size,     # kernel size
            stride=stride,               # stride
            padding=1,                   # padding
            groups=in_ch                 # the number of input channels is equal to the number of groups, and each input channel is assigned a convolution kernel
        )
 
        # Pointwise Convolution
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,           # input channels =output channels of depthwise convolution
            out_channels=out_ch,         # output channels=classes
            kernel_size=1,               # kernel size=1
            stride=1,                    # stride
            padding=0,                   # no padding
            groups=1                     # the number of input channels is equal to the number of groups, and each input channel has a convolution kernel
        )
 
    def forward(self, input):
        # depthwise Convolution
        out = self.depth_conv(input)
        # pointwise Convolution
        out = self.point_conv(out)
        return out
    
