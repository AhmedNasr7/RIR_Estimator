import torch
from torch import nn
import torch.nn.functional as F
import numpy as np 
from watermark import watermark
from dataclasses import dataclass
import random

from torch.utils.tensorboard import SummaryWriter
from dataset import DataLoaderWrapper
from models.RIR_Model import RIR_Estimator
from models.configs import EncoderParams, DecoderParams
from losses import *
from utils import *
from loguru import logger

@dataclass
class train_configs:
    batch_size = 128
    train_ratio = 0.85
    
    epochs = 200
    learning_rate = 8e-5
    learning_rate_decay_factor = 0.7
    decay_period = 40

    data_dir = "./data/"
    audio_dir = data_dir + "/audio_dir/"
    rir_dir = data_dir + "/rir_dir/"

def compute_loss(y_hat, y):
    return 0.5 * EDR_Loss()(y_hat, y) + 0.5 * MSE_Loss()(y_hat, y)


def train(train_configs):

    torch.manual_seed(0) 
    np.random.seed(0)
    random.seed(0)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    model = RIR_Estimator(EncoderParams, DecoderParams).to(device)

    model = torch.compile(model)

    dataloader = DataLoaderWrapper(train_configs.audio_dir, train_configs.rir_dir, 
                                   train_ratio=train_configs.train_ratio, batch_size=train_configs.batch_size)

    train_loader = dataloader.train_dataloader()
    val_loader = dataloader.val_dataloader()

    # criterion = ModelTrainer.criterion()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=train_configs.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=train_configs.decay_period, gamma=train_configs.learning_rate_decay_factor)  # Adjust as needed



    # Set up TensorBoard writer
    writer = SummaryWriter()

    # Training loop
    for epoch in range(train_configs.epochs):  # Adjust as needed
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data

            inputs = copy_to_device(inputs)
            labels = copy_to_device(labels)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = compute_loss(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if i % 100 == 99:  # Print every 100 mini-batches
                logger.info(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 100:.3f}")
                running_loss = 0.0

        # Log training loss to TensorBoard
        writer.add_scalar('training loss', running_loss, epoch)
        scheduler.step()


        # Validation loop
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                inputs, labels = data

                inputs = copy_to_device(inputs)
                labels = copy_to_device(labels)

                outputs = model(inputs)
                loss = compute_loss(outputs, labels)
                val_loss += loss.item()

        # Log validation loss to TensorBoard
        writer.add_scalar('validation loss', val_loss, epoch)

        print(f"Epoch {epoch + 1}, Training loss: {running_loss:.3f}, Validation loss: {val_loss:.3f}")

    print("Finished Training")

    # Close TensorBoard writer
    writer.close()




if __name__ == "__main__":

    print(watermark(packages="numpy,torch,lightning", python=True))

    train(train_configs)