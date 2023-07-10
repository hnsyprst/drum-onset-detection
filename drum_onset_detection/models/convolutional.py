import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

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
    

def init_kernels(win_len, win_inc, fft_len, win_type=None, invers=False):
    if win_type == 'None' or win_type is None:
        window = np.ones(win_len)
    else:
        window = get_window(win_type, win_len, fftbins=True)#**0.5
    
    fourier_basis = np.fft.rfft(np.eye(fft_len))[:win_len]
    kernel = fourier_basis.T
    
    if invers :
        kernel = np.linalg.pinv(kernel).T 

    kernel = kernel*window
    return torch.from_numpy(kernel.astype(np.float32))


class ConvSTFT(nn.Module):
    def __init__(self, win_len, win_inc, fft_len=None, win_type='hamming'):
        super(ConvSTFT, self).__init__() 
        
        if fft_len == None:
            self.fft_len = np.int(2**np.ceil(np.log2(win_len)))
        else:
            self.fft_len = fft_len
        
        kernel = init_kernels(win_len, win_inc, self.fft_len, win_type)
        self.register_buffer('weight', kernel.unsqueeze(1).unsqueeze(1))
        self.stride = win_inc
        self.win_len = win_len
        self.dim = self.fft_len

    def forward(self, inputs):
        inputs = F.pad(inputs,[self.win_len-self.stride, self.win_len-self.stride])
        # Somehow make this work with 2d input
        outputs = F.conv2d(inputs, self.weight, stride=(1, self.stride))
        return outputs


class AutoConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.AutoConv3d = (nn.Sequential(nn.Conv3d(in_channels=in_channels,
                                                    out_channels=out_channels,
                                                    kernel_size=kernel_size,
                                                    stride=stride,
                                                    padding=(0, stride[0]//2, stride[1]//2)),
                                        nn.ReLU(inplace=True)
        ))
    
    def forward(self, input):
        return self.AutoConv3d(input)
    

class TFDConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.stft = ConvSTFT(512, 512 // 4, 512, 'hann')
        self.conv1 = AutoConv3d(in_channels=1,  out_channels=16, kernel_size=(1, 3, 3), stride=(1, 3, 3)) # out = 918, 170, 16
        self.conv2 = AutoConv3d(in_channels=16, out_channels=32, kernel_size=(1, 3, 3), stride=(1, 2, 2)) #       458, 84,  32
        self.conv3 = AutoConv3d(in_channels=32, out_channels=64, kernel_size=(1, 3, 3), stride=(1, 3, 3)) #       152, 27,  64
        #self.conv4 = AutoConv3d(in_channels=64, out_channels=64, kernel_size=(1, 5, 5), stride=(1, 3, 3)) #       50,  9,   64
        #self.conv5 = AutoConv2d(in_channels=64, out_channels=64, kernel_size=(7, 1), stride=(2, 1)) #      23,  2,   64
        #self.pool1 = nn.AdaptiveAvgPool2d((1, 32)) # 64, 1, 32
        self.dense = nn.Linear(in_features=896, out_features=5) # Fixed for frame_len = 512, fft_len = 512 TODO: Unfix 
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.stft(input)
        output = output.transpose(1, 2)
        output = output.unsqueeze(1)
        output = self.conv1(output)
        output = self.conv2(output)
        output = self.conv3(output)
        #output = self.conv4(output)
        #output = self.conv5(output)
        #output = self.pool1(output)
        output = output.transpose(1, 2)
        output = torch.flatten(output, start_dim=2)
        output = self.dense(output)
        #output = self.relu(output)
        return output
    

class miniMobileNet(nn.Module):
    