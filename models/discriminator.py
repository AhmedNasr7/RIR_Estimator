import torch
import torch.nn as nn

from loguru import logger


def old_conv3x1(in_planes, out_planes, stride=1):
    "3x1 convolution with padding"
    kernel_length  = 3
    return nn.Conv1d(in_planes, out_planes, kernel_size=kernel_length, stride=stride,
                     padding=1, bias=False)

class D_GET_LOGITS(nn.Module): #not chnaged yet
    def __init__(self, ndf, nef, bcondition=False):
        super(D_GET_LOGITS, self).__init__()
        self.df_dim = ndf
        self.ef_dim = nef
        self.bcondition = bcondition
        kernel_length =41
        if bcondition:
            self.convd1d =  nn.ConvTranspose1d(ndf*8,ndf //2,kernel_size=kernel_length,stride=1, padding=20)

            self.outlogits = nn.Sequential(
                old_conv3x1(ndf //2 + nef, ndf //2 ),
                # nn.BatchNorm1d(ndf // 2 ),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv1d(ndf //2 , 1, kernel_size=16, stride=4),
                nn.Sigmoid()
                )
        else:
  
            self.convd1d =  nn.ConvTranspose1d(ndf*8,ndf //2,kernel_size=kernel_length,stride=1, padding=20)
            self.outlogits = nn.Sequential(
                nn.Conv1d(ndf // 2 , 1, kernel_size=16, stride=4),
                nn.Sigmoid())

    def forward(self, h_code, c_code=None):
        # conditioning output
        h_code = self.convd1d(h_code)
        if self.bcondition and c_code is not None:
            #print("mode c_code1 ",c_code.size())
            c_code = c_code.view(-1, self.ef_dim, 1)
            #print("mode c_code2 ",c_code.size())

            c_code = c_code.repeat(1, 1, 16)
            # state size (ngf+egf) x 16
            #print("mode c_code ",c_code.size())
            #print("mode h_code ",h_code.size())

            h_c_code = torch.cat((h_code, c_code), 1)
        else:
            h_c_code = h_code

        output = self.outlogits(h_c_code)

        return output.view(-1)


# #Adapted from Fast-RIR paper:

class STAGE1_Discriminator(nn.Module):
    def __init__(self):
        super(STAGE1_Discriminator, self).__init__()
        self.df_dim = 64
        self.ef_dim = 128

        ndf, nef = self.df_dim, self.ef_dim
        kernel_length =41

        self.encode_RIR = nn.Sequential(
            nn.Conv1d(1, ndf, kernel_length, 4, 20, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 1024
            nn.Conv1d(ndf, ndf * 2, kernel_length, 4, 20, bias=False),
            # nn.BatchNorm1d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size (ndf*2) x 256
            nn.Conv1d(ndf*2, ndf * 4, kernel_length, 4, 20, bias=False),
            # nn.BatchNorm1d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # # state size (ndf*4) x 64
            nn.Conv1d(ndf*4, ndf * 8, kernel_length, 4, 20, bias=False),
            # nn.BatchNorm1d(ndf * 8),
            # state size (ndf * 8) x 16)
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.get_cond_logits = D_GET_LOGITS(ndf , nef)
        # self.get_uncond_logits = None

    def forward(self, RIRs):
        #print("model RIRs ",RIRs.size())
        RIR_embedding = self.encode_RIR(RIRs)
        output = self.get_cond_logits(RIR_embedding)
        #print("models RIR_embedding ",RIR_embedding.size())

        return output


if __name__ == "__main__":


    L_in =  2330

    
    disc = STAGE1_Discriminator()
    
    x = torch.randn(1, 1, 2330)

    disc_output = disc(x)

    logger.info(f"disc output: {disc_output.shape}")

