from dataclasses import dataclass

@dataclass
class EncoderParams:

    input_length =  44_100
    input_channels = 1
    channels = [512, 1024]
    lengths = [512, 64]
    kernel_sizes = [22050, 256]
    strides = [64, 8]
    paddings = [5340, 124]
    use_bn = [False, True]




@dataclass
class DecoderParams:
    input_length = 64
    lengths = [134, 270, 582, 1164, 2330]
    input_channels = 1024
    channels=[1024, 512, 256, 128, 64] 
    kernel_sizes=[5, 3, 41, 7, 5]
    strides=[2, 2, 2, 2, 2]
    paddings=[1, 1 , 0, 21, 1]
    output_paddings=[1, 1, 1, 1, 1]
