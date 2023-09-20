import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from loguru import logger 
from models.discriminator import STAGE1_Discriminator

class EDR_Loss(nn.Module):
    def __init__(self, n_fft=512, hop_length=128, device="cuda"):
        super(EDR_Loss, self).__init__()

        self.fs = 44_100
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
                        ).to(torch.device(device))
        
        self.center_frequencies = [16, 32, 64, 128, 256, 512, 1000, 2000, 4000] # octav freqs from 16 hz to 4k hz as described by paper


    def compute_stft(self, x: torch.Tensor):
        spec = self.stft_func(x)
        return  torch.permute(spec, (0, 1, 3, 2))





    def compute_energy_bands(self, S):
        """
        Compute the energy remaining in a set of octave frequency bands at time t.
        
        Args:
            S (torch.Tensor): Spectrogram of the RIR with shape (b, c, T, F), 
                            where T is the total number of time frames and F is the 
                            number of frequency bins.
            center_frequencies (list): List of center frequencies (in Hz) for the bands.
            t (int): Time frame index.
            
        Returns:
            sum_energy_bands (float): sum of energy values for each band.
        """
        F = S.shape[3]

        nyquist_f = self.fs  / 2

        bin_indices = torch.tensor([int(f * F / nyquist_f) for f in self.center_frequencies])

  
        # Compute the total energy in the frequency bin
        total_energy_bin = torch.sum(torch.abs(S[:, :, :, bin_indices])**2, dim=2)
            
            
        
        return total_energy_bin
           
    

    def forward(self, x, y):

        # Compute STFT of the input signals

        spec1 = self.compute_stft(x) # torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length, return_complex=False, center=True)
        spec2 = self.compute_stft(y) # torch.stft(y, n_fft=self.n_fft, hop_length=self.hop_length, return_complex=False, center=True)

        
        # logger.debug(f"tensor shapes: {x.shape} {y.shape}")
  
        # logger.debug(f"spec shapes: {spec1.shape} {spec2.shape}")

        edr1 = self.compute_energy_bands(spec1)
        edr2 = self.compute_energy_bands(spec2)

        # logger.debug(f"edr shapes: {edr1.shape} {edr2.shape}")

        # Compute mean squared error loss between the squared magnitude EDRs

        mse = F.mse_loss(edr1, edr2)
    
        return mse


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

    x = torch.randn(1, 1, 11_025).cuda()
    y = torch.randn(1, 1, 11_025).cuda()

    criterion = EDR_Loss()

    loss = criterion(x, y)



    # disc = STAGE1_Discriminator()

    # d = disc(x)

    # criterion = CGAN_Loss()

    # loss = criterion(d)

    logger.info(f"loss: {loss}")

