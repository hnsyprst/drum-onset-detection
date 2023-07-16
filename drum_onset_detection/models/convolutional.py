import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchlibrosa as tl

from scipy.signal import get_window
#from stream import Stream

class AutoConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.AutoConv2d = (nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                    out_channels=out_channels,
                                                    kernel_size=kernel_size,
                                                    stride=stride,
                                                    padding=(stride[0]//2, 0)),
                                        nn.ReLU(inplace=True)
        ))
    
    def forward(self, input):
        return self.AutoConv2d(input)


class TDConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = AutoConv2d(in_channels=1,  out_channels=16, kernel_size=(1, 7), stride=(1, 3)) # out = 918, 170, 16
        self.conv2 = AutoConv2d(in_channels=16, out_channels=32, kernel_size=(1, 5), stride=(1, 2)) #       458, 84,  32
        self.conv3 = AutoConv2d(in_channels=32, out_channels=64, kernel_size=(1, 7), stride=(1, 3)) #       152, 27,  64
        self.conv4 = AutoConv2d(in_channels=64, out_channels=64, kernel_size=(1, 5), stride=(1, 3)) #       50,  9,   64
        #self.conv5 = AutoConv2d(in_channels=64, out_channels=64, kernel_size=(7, 1), stride=(2, 1)) #      23,  2,   64
        #self.pool1 = nn.AdaptiveAvgPool2d((1, 32)) # 64, 1, 32
        self.dense = nn.Linear(in_features=512, out_features=5) # Fixed for frame_len = 512 TODO: Unfix 
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        #output = self.conv5(output)
        #output = self.pool1(output)
        output = output.transpose(1, 2)
        output = torch.flatten(output, start_dim=2)
        output = self.dense(output)
        #output = self.relu(output)
        return output


class GlobalAveragePool2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input.mean(dim = (-2, -1))
    

class TFDConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stft = tl.Spectrogram(512, 512 // 4, 512)
        self.conv1 = AutoConv2d(in_channels=1,  out_channels=16, kernel_size=(3, 3), stride=(3, 3)) # out = 918, 170, 16
        self.conv2 = AutoConv2d(in_channels=16, out_channels=32, kernel_size=(3, 3), stride=(2, 2)) #       458, 84,  32
        self.conv3 = AutoConv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(3, 3)) #       152, 27,  64
        self.pool = GlobalAveragePool2d()
        self.dense = nn.Linear(in_features=64, out_features=5) # Fixed for frame_len = 512, fft_len = 512 TODO: Unfix 
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.stft(input)
        #output_mag = torch.abs(output) # 8, 1, 5, 257
        output = self.conv1(output)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.pool(output)
        output = self.dense(output)
        #output = self.relu(output)
        return output
    

class miniMobileNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.stft_extractor = tl.Spectrogram(512, 512 // 10, 512)

        self.model = nn.Sequential(
            self.conv_block(1, 3, 2),

            self.conv_block(3, 32, 2),
            self.depthwise_conv_block(32, 64, 1),
            self.depthwise_conv_block(64, 128, 2),
            self.depthwise_conv_block(128, 128, 1),
            self.depthwise_conv_block(128, 256, 2),
            self.depthwise_conv_block(256, 256, 1),
            self.depthwise_conv_block(256, 512, 2),
            self.depthwise_conv_block(512, 512, 1),
            GlobalAveragePool2d()
        )
        self.labeler = nn.Linear(in_features=512, out_features=5)


    @staticmethod
    def conv_block(in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    @staticmethod
    def depthwise_conv_block(in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    

    def forward(self, input):
        mag_spec = self.stft_extractor(input)
        features = self.model(mag_spec)
        labels = self.labeler(features)
        return labels