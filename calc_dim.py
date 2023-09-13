import numpy as np

L_in =  512
input_channels = 512
output_channels = 1024
kernel_size = 256
stride = 8
padding = 124
dilation = 1


#########################


# L_in_up = 1164
# stride_up  = 2 # 4 
# padding_up = 0
# dilation_up = 1
# kernel_size_up = 3
# out_padding = 1 # 4


L_in_up = 2330
stride_up  = 1 # 4 
padding_up = 20
dilation_up = 1
kernel_size_up = 41
out_padding =  0


### 
L_out = np.floor((L_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1) 


###
L_out_up = (L_in_up - 1) * stride_up - 2 * padding_up + dilation_up * (kernel_size_up - 1) + out_padding + 1


#######################

print("L_out conv2d: ", L_out) 

print("L_out_up: ", L_out_up)


""""
1: 

64 ---> 134

L_in_up = 64
stride_up  = 2 # 4 
padding_up = 1
dilation_up = 1
kernel_size_up = 5
 # 4 
out_padding = 5 # 4


2: 
134 --> 270
L_in_up = 134
stride_up  = 2 # 4 
padding_up = 1
dilation_up = 1
kernel_size_up = 3
 # 4 
out_padding = 3 # 4


3: 

270 ---> 582


L_in_up = 270
stride_up  = 2 # 4 
padding_up = 0
dilation_up = 2
kernel_size_up = 7
out_padding =  31




582 ---> 1164

L_in_up = 582
stride_up  = 2 # 4 
padding_up = 2
dilation_up = 1
kernel_size_up = 5
 # 4 
out_padding = 1 # 4


1164 ---> 2330

5: 

L_in_up = 1164
stride_up  = 2 # 4 
padding_up = 0
dilation_up = 1
kernel_size_up = 3
 # 4 
out_padding = 1 # 4

target_len = 2330
"""

# [1, 1 , 0, 2, 0]