from dataclasses import dataclass

@dataclass
class params:

    input_length =  44_100
    input_channels = 1
    channels = [512, 1024]
    lengths = [512, 64]
    kernel_sizes = [22050, 256]
    strides = [64, 8]
    paddings = [5340, 124]
    use_bn = [False, True]


