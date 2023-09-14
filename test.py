import numpy as np

# path = "./data/IRs_HL00W.mat"


# with h5py.File(path, 'r') as file:
#     my_array = np.array(file["data"][:])
#     # min_, max_= np.min(my_array), np.max(my_array)

#     print(my_array.shape)

import torch.nn as nn
import torch

import torch.nn as nn


## test and experiments for the decoder design to estimate the needed samples

class TransposeConvolutionDecoder(nn.Module):
    def __init__(self):
        super(TransposeConvolutionDecoder, self).__init__()
        
        # Define 5 transposed convolution layers
        self.conv1 = nn.ConvTranspose1d(in_channels=1024, out_channels=512, kernel_size=5, stride=2, padding=16, output_padding=0, dilation=1)
        self.conv2 = nn.ConvTranspose1d(in_channels=512, out_channels=256, kernel_size=5, stride=2, padding=32, output_padding=0, dilation=2)
        self.conv3 = nn.ConvTranspose1d(in_channels=256, out_channels=128, kernel_size=7, stride=4, padding=74, output_padding=0, dilation=2)
        self.conv4 = nn.ConvTranspose1d(in_channels=128, out_channels=64, kernel_size=7, stride=4, padding=141, output_padding=0, dilation=1)
        self.conv5 = nn.ConvTranspose1d(in_channels=64, out_channels=1, kernel_size=5, stride=2, padding=6, output_padding=0, dilation=1)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x

# Initialize the decoder
decoder = TransposeConvolutionDecoder()

# Input shape: (batch_size, channels, length)
# Output shape: (batch_size, channels, length)
# Example usage:
import torch

input_tensor = torch.randn(1, 1024, 128)
output_tensor = decoder(input_tensor)

print("Output shape:", output_tensor.shape)
