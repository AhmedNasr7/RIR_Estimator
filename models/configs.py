from dataclasses import dataclass

@dataclass
class EncoderParams:

    input_length =  44_100
    input_channels = 1
    channels = [512, 1024]
    lengths = [512, 128]
    kernel_sizes = [22050, 256]
    strides = [64, 4]
    paddings = [5340, 127]
    use_bn = [False, True]




@dataclass
class DecoderParams:
    input_length = 64
    lengths = [134, 270, 582, 1164, 2330]
    input_channels = 1024
    channels=[1024, 512, 256, 128, 64] 
    kernel_sizes= [5, 5, 7, 7, 5] #[5, 3, 41, 7, 5]
    strides=[2, 2, 4, 4, 2]
    paddings=[16, 32 , 74, 141, 6]
    output_paddings=[0, 0, 0, 0, 0]
    dilation = [1, 2, 2, 1, 1]
