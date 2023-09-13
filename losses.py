import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from loguru import logger 
from models.discriminator import STAGE1_Discriminator

class EDR_Loss(nn.Module):
    def __init__(self, n_fft=2048, hop_length=512):
        super(EDR_Loss, self).__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        win_length = n_fft

        self.stft_func = T.Spectrogram(
                            n_fft=n_fft,
                            win_length=win_length,
                            hop_length=hop_length,
                            center=True,
                            pad_mode="reflect",
                            power=2.0,
                        )

    def forward(self, x, y):
        # Compute STFT of the input signals
        spec1 = self.stft_func(x) # torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, return_complex=False, center=True)
        spec2 = self.stft_func(y) # torch.stft(y, n_fft=self.n_fft, hop_length=self.hop_length, return_complex=False, center=True)
            
        # logger.debug(f"spec shapes: {spec1.shape} {spec2.shape}")

        # Compute mean squared error loss between the squared magnitude spectrograms
        mse_loss = F.mse_loss(spec1, spec2)
    
        return mse_loss



class MSE_Loss(nn.Module):
    def __init__(self):
        super(MSE_Loss, self).__init__()

    def forward(self, x, y):

        mse_loss = F.mse_loss(x, y)
    
        return mse_loss



class CGAN_Loss(nn.Module):
    def __init__(self):
        super(CGAN_Loss, self).__init__()

    def forward(self, x):

        loss = torch.mean(torch.log(1 - x))
    
        return loss


if __name__ == "__main__":

    x = torch.randn(1, 1, 2330)
    y = torch.randn(1, 1, 2330)

    disc = STAGE1_Discriminator()

    d = disc(x)

    criterion = CGAN_Loss()

    loss = criterion(d)

    logger.info(f"loss: {loss}")

