import torch
from torch.utils.data import Dataset, DataLoader
import os
import h5py
import numpy as np
import torchaudio 
import lightning as L
from glob import glob


class SR_Dataset(Dataset):
    def __init__(self, audio_dir, rir_dir):
        self.audio_dir = audio_dir
        self.rir_dir = rir_dir
        self.audio_files = glob(audio_dir + "/*.wav")
        self.rir_files = glob(rir_dir + "/*.mat")

        self.audio_files.sort()
        self.rir_files.sort()

        assert len(self.audio_files) == len(self.mat_files), "Number of audio and mat files must match"
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        rir_file = self.rir_files[idx]

        
        # Load audio data
        audio_data, sr = torchaudio.load(audio_file).float()
        
        # Load mat data using h5py
        with h5py.File(rir_file, 'r') as f:
            rir_numpy = np.array(f['data'][:])[-1]  


        rir_data = torch.from_numpy(rir_numpy).float().unsqueeze(0)
        
        return audio_data, rir_data





class LitDataModule(L.LightningDataModule):
    def __init__(self, audio_dir, rir_dir, batch_size=128):
        super().__init__()
        self.audio_dir = audio_dir
        self.rir_dir = rir_dir
        self.batch_size = batch_size
    
    def setup(self, stage=None):
        self.dataset = SR_Dataset(audio_dir=self.audio_dir, rir_dir=self.rir_dir)

    def train_dataloader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, drop_last=False, num_workers=4)



if __name__ == "__main__":

    # Usage example:
    audio_dir = '/path/to/audio/files'
    mat_dir = '/path/to/mat/files'


    dataset = SR_Dataset(audio_dir, mat_dir)

    # Example of how to get a sample
    sample_audio, sample_mat = dataset[0]
