import torch
from torch import nn
import torch.nn.functional as F
import numpy as np 
import lightning as L
from watermark import watermark
from dataclasses import dataclass


from dataset import LitDataModule
from models.RIR_Model import RIR_Estimator
from LitModule import LitModel
from models.configs import EncoderParams, DecoderParams
from utils import PeriodicLRDecayCallback


@dataclass
class train_configs:
    batch_size = 128
    
    epochs = 200
    learning_rate = 8e-5
    learning_rate_decay_factor = 0.7
    decay_period = 40

    data_dir = "./data/"
    audio_dir = data_dir + "/audio_dir/"
    rir_dir = data_dir + "/rir_dir/"



def train(train_configs):

    L.seed_everything(42) # for reproducibility 


    model = RIR_Estimator(EncoderParams, DecoderParams)

    model = torch.compile(model)

    lightning_model = LitModel(model=model, learning_rate=train_configs.epochs) # change lit module


    data_module = LitDataModule(audio_dir=train_configs.audio_dir, rir_dir=train_configs.rir_dir, batch_size=train_configs.batch_size)
    
    
    trainer = L.Trainer(
            max_epochs=train_configs.epochs, accelerator="cuda", devices="auto", deterministic=True, 
            callbacks=[PeriodicLRDecayCallback(train_configs.decay_period, train_configs.learning_rate_decay_factor)]
            )

    trainer.fit(model=lightning_model, datamodule=data_module)





if __name__ == "__main__":

    print(watermark(packages="numpy,torch,lightning", python=True))

    train(train_configs)