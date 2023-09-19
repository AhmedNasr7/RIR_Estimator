import torch
from torch.utils.data import Dataset, DataLoader, random_split
import os
import numpy as np
import torchaudio 
from glob import glob


class SR_Dataset(Dataset):
    def __init__(self, audio_dir, rir_dir):
        self.audio_dir = audio_dir
        self.rir_dir = rir_dir
        self.audio_files = glob(audio_dir + "/*.wav")
        self.rir_files = glob(rir_dir + "/*.npy")

        self.audio_files.sort()
        self.rir_files.sort()

        self.fs = 44_100

        self.rir_len = self.fs * 0.25

        assert len(self.audio_files) == len(self.rir_files), "Number of audio and mat files must match"
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        rir_file = self.rir_files[idx]

        
        # Load audio data
        audio_data, sr = torchaudio.load(audio_file).float()

        
        rir_numpy = np.load(rir_file)
        # # Load mat data using h5py
        # with h5py.File(rir_file, 'r') as f:
        #     rir_numpy = np.array(f['data'][:])[-1]  


        rir_data = torch.from_numpy(rir_numpy).float().unsqueeze(0)
        
        
        return audio_data, rir_data





class DataLoaderWrapper:
    def __init__(self, audio_dir, rir_dir, train_ratio=0.85, batch_size=128):
        super().__init__()
        self.audio_dir = audio_dir
        self.rir_dir = rir_dir
        self.batch_size = batch_size

        self.dataset = SR_Dataset(audio_dir=self.audio_dir, rir_dir=self.rir_dir)

        # Split dataset into training and validation sets
        train_size = int(train_ratio * len(self.dataset))

        val_size = len(dataset) - train_size

        self.train_dataset, self.val_dataset = random_split(dataset, [train_size, val_size])
        

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False, num_workers=4, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size // 4, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)


if __name__ == "__main__":

    # Usage example:
    audio_dir = '/path/to/audio/files'
    rir_dir = '/path/to/mat/files'


    dataset = SR_Dataset(audio_dir, rir_dir)

    # Example of how to get a sample
    sample_audio, sample_mat = dataset[0]
