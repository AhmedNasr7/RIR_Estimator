import torch
import torch.nn as nn

from loguru import logger


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, use_bn=False):
        super(ConvBlock, self).__init__()

        self.use_bn = use_bn
        
        # Create a module list of 1D convolutional layers

        self.conv1d = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding) 
        if self.use_bn:
            self.bn = torch.nn.BatchNorm1d(out_channels)
        
        # Define a Leaky ReLU activation function with a negative slope of 0.2
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor):

        x = self.conv1d(x)
        
        if self.use_bn:
            x = self.bn(x)

        x = self.act(x)

        return x



class UpConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, dilation=1, use_bn=True):
        super(UpConvBlock, self).__init__()

        self.use_bn = use_bn

        self.tr_conv1d = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, output_padding=output_padding, dilation=dilation)

        if self.use_bn:
            self.bn = torch.nn.BatchNorm1d(out_channels)
        
        # Define a PReLU activation function
        self.act = nn.PReLU()

    def forward(self, x: torch.Tensor):

        x = self.tr_conv1d(x)
        
        if self.use_bn:
            x = self.bn(x)

        x = self.act(x)

        return x


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        self.config = config

        self.conv_1 = ConvBlock(in_channels=self.config.input_channels, out_channels=self.config.channels[0], \
                               kernel_size=self.config.kernel_sizes[0], stride=self.config.strides[0], \
                               padding=self.config.paddings[0], use_bn=self.config.use_bn[0])
        
        self.conv_2 = ConvBlock(in_channels=self.config.channels[0], out_channels=self.config.channels[1], \
                               kernel_size=self.config.kernel_sizes[1], stride=self.config.strides[1], \
                               padding=self.config.paddings[1], use_bn=self.config.use_bn[1])
        

    def forward(self, x: torch.Tensor):

       x = self.conv_1(x)

       x = self.conv_2(x)

       return x
    




class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        self.config = config

        self.up_conv_1 = UpConvBlock(in_channels=self.config.input_channels, \
                                    out_channels=self.config.channels[0], 
                                    kernel_size=self.config.kernel_sizes[0], \
                                    stride=self.config.strides[0], padding=self.config.paddings[0], \
                                    output_padding=self.config.output_paddings[0], dilation=self.config.dilation[0])
        
        self.up_conv_2 = UpConvBlock(in_channels=self.config.channels[0], \
                                    out_channels=self.config.channels[1], 
                                    kernel_size=self.config.kernel_sizes[1], \
                                    stride=self.config.strides[1], padding=self.config.paddings[1], \
                                    output_padding=self.config.output_paddings[1], dilation=self.config.dilation[1])

 
        self.up_conv_3 = UpConvBlock(in_channels=self.config.channels[1], \
                                    out_channels=self.config.channels[2], 
                                    kernel_size=self.config.kernel_sizes[2], \
                                    stride=self.config.strides[2], padding=self.config.paddings[2], \
                                    output_padding=self.config.output_paddings[2], dilation=self.config.dilation[2])

 
        self.up_conv_4 = UpConvBlock(in_channels=self.config.channels[2], \
                                    out_channels=self.config.channels[3], 
                                    kernel_size=self.config.kernel_sizes[3], \
                                    stride=self.config.strides[3], padding=self.config.paddings[3], \
                                    output_padding=self.config.output_paddings[3], dilation=self.config.dilation[3])

 
        self.up_conv_5 = UpConvBlock(in_channels=self.config.channels[3], \
                                    out_channels=self.config.channels[4], 
                                    kernel_size=self.config.kernel_sizes[4], \
                                    stride=self.config.strides[4], padding=self.config.paddings[4], \
                                    output_padding=self.config.output_paddings[4], dilation=self.config.dilation[4])

        self.last_trConv1d = nn.ConvTranspose1d(in_channels=64, out_channels=1, \
                                                 kernel_size=3, stride=1, padding=1, output_padding=0)

        self.act = nn.Tanh()

    def forward(self, x: torch.Tensor):
       
       x = self.up_conv_1(x)
       x = self.up_conv_2(x)
       x = self.up_conv_3(x)
       x = self.up_conv_4(x)
       x = self.up_conv_5(x)

       x = self.last_trConv1d(x)

       self.act(x)

       return x
    


class RIR_Estimator(nn.Module):
    def __init__(self, encoder_params, decoder_params):
        super(RIR_Estimator, self).__init__()

        self.encoder = Encoder(encoder_params)
        self.decoder = Decoder(decoder_params)

        

    def forward(self, x: torch.Tensor):
       
       x = self.encoder(x)
       x = self.decoder(x)

       return x
    



if __name__ == "__main__":

    from configs import EncoderParams, DecoderParams



    # test_up = UpConvBlock(in_channels=1024, \
    #                     out_channels=512, \
    #                     kernel_size=5, \
    #                     stride=2, \
    #                     padding=511, \
    #                     output_padding=1, dilation=1)
    

    # x = torch.randn(1, 1024, 6390)

    # y = test_up(x)

    # logger.info(f"up conv output shape: {y.shape}")


    ## Tests

    L_in =  44_100
    input_channels = 1
    output_channels = 512
    kernel_size = 22050
    stride = 64
    padding = 5340
    dilation = 1
    
    encoder = Encoder(EncoderParams)
    
    x = torch.randn(1, 1, L_in)

    encoder_output = encoder(x)

    logger.info(f"encoder output: {encoder_output.shape}")

    decoder = Decoder(DecoderParams)

    decoder_output = decoder(encoder_output)

    logger.info(f"decoder output shape: {decoder_output.shape}")



    ## Testing Decoder: 

    model = RIR_Estimator(EncoderParams, DecoderParams)

    rir = model(x)

    logger.info(f"rir shape: {rir.shape}") 





    # up_conv = UpConvBlock(in_channels=output_data.shape[1], out_channels=512, kernel_size=3, \
    #                       stride=2, padding=0, output_padding=0)


    # up_out = up_conv(output_data)

    # logger.info(f"{up_out.shape}")


