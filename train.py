import torch
from torch import nn
import torch.nn.functional as F
import numpy as np 
from watermark import watermark
from dataclasses import dataclass
import random

# from torch.utils.tensorboard import SummaryWriter
from dataset import DataLoaderWrapper
from models.RIR_Model import RIR_Estimator
from models.configs import EncoderParams, DecoderParams
from losses import *
from utils import *
from loguru import logger
from tqdm import tqdm 
import warnings

warnings.filterwarnings("ignore")


@dataclass
class train_configs:
    batch_size = 256 # paper says 128  
    train_ratio = 0.85
    
    epochs = 200
    learning_rate = 8e-5
    learning_rate_decay_factor = 0.7
    decay_period = 40

    data_dir = "../dataProcessing/"
    audio_dir = data_dir + "processed_wavs2_1sec"
    rir_dir = data_dir + "RIRs/"

def compute_loss(y_hat, y):
    return 0.5 * EDR_Loss()(y_hat, y) + 0.5 * MSE_Loss()(y_hat, y)


def train(train_configs):

    torch.manual_seed(0) 
    np.random.seed(0)
    random.seed(0)

    best_loss = float("inf")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


    model = RIR_Estimator(EncoderParams, DecoderParams).to(device)

    model = torch.compile(model)

    dataloader = DataLoaderWrapper(train_configs.audio_dir, train_configs.rir_dir, 
                                   train_ratio=train_configs.train_ratio, batch_size=train_configs.batch_size)

    train_loader = dataloader.train_dataloader()
    val_loader = dataloader.val_dataloader()

    # criterion = ModelTrainer.criterion()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=train_configs.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=train_configs.decay_period * len(train_loader), gamma=train_configs.learning_rate_decay_factor)  # Adjust as needed



    # Set up TensorBoard writer
    # writer = SummaryWriter()

    # Training loop

    max_epochs = train_configs.epochs
    for epoch in range(max_epochs):  # Adjust as needed
        running_loss = 0.0

        with tqdm(train_loader, desc=f"epoch={epoch + 1}/{max_epochs}") as pbar:
            for i, data in enumerate(pbar):
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
                    logger.info(f"[Epoch: {epoch + 1}, minibatch: {i + 1}] average loss: {running_loss / 100:.3f}")
                    pbar.set_postfix(loss=running_loss/100)
                    running_loss = 0.0

                # Log training loss to TensorBoard
                # writer.add_scalar('training loss', running_loss, epoch)
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
        # writer.add_scalar('validation loss', val_loss, epoch)

        # val_loss = val_loss / len(val_loader)

        logger.info(f"Epoch {epoch + 1}, Training loss: {running_loss:.3f}, Validation loss: {val_loss:.3f}")

        if val_loss < best_loss:
            best_loss = val_loss

            torch.save(model.state_dict(), f"best.pth")
            

    print("Finished Training")

    # Close TensorBoard writer
    # writer.close()




if __name__ == "__main__":

    print(watermark(packages="numpy,torch,lightning", python=True))

    train(train_configs)