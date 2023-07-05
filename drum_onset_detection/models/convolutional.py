import torch
import torch.nn as nn
#from stream import Stream

class AutoConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.AutoConv2d = (nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                    out_channels=out_channels,
                                                    kernel_size=kernel_size,
                                                    stride=stride,
                                                    padding=stride//2),
                                        nn.ReLU(inplace=True)
        ))
    
    def forward(self, input):
        return self.AutoConv2d(input)


class StreamingConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = AutoConv2d(in_channels=1, out_channels=16, kernel_size=7, stride=3) # out = 918 x 170 x 16
        self.conv2 = AutoConv2d(in_channels=16, out_channels=32, kernel_size=5, stride=2) #      458 x 84 x 32
        self.conv3 = AutoConv2d(in_channels=32, out_channels=64, kernel_size=7, stride=3) #      152 x 27 x 64
        self.conv4 = AutoConv2d(in_channels=64, out_channels=64, kernel_size=5, stride=3) #      50 x 9 x 64
        self.conv5 = AutoConv2d(in_channels=64, out_channels=64, kernel_size=7, stride=2) #      23 x 2 x 64
        self.dense = nn.Linear(in_features=2944, out_features=5)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = torch.flatten(output)
        output = self.dense(output)
        output = self.relu(output)
        return output