import lightning as L
import torch
import torch.nn.functional as F
import torchmetrics
from torch.utils.data.dataset import random_split
from losses import *


class LitModel(L.LightningModule):
    def __init__(self, model, learning_rate):
        super().__init__()

        self.learning_rate = learning_rate
        self.mse_weight = 0.5

        self.model = model

        # Save settings and hyperparameters to the log directory
        # but skip the model parameters
        self.save_hyperparameters(ignore=["model"])
        self.edr_loss = EDR_Loss()
        self.mse_loss = MSE_Loss()
        self.cgan_loss = CGAN_Loss()
        

    def criterion(self, x, y):
        loss = self.edr_loss(x, y) + self.mse_weight * self.mse_loss(x, y) # not adding cgan?

        return loss
        

    def forward(self, x):
        return self.model(x)

    def _forward_step(self, batch):
        features, targets = batch
        preds = self(features)

        loss = self.criterion(preds, targets)

        return loss, targets, preds

    def training_step(self, batch, batch_idx):
        loss, targets, preds = self._shared_step(batch)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True, on_step=True)
      
        return loss

    def validation_step(self, batch, batch_idx):
        loss, targets, preds = self._shared_step(batch)

        self.log("val_loss", loss, prog_bar=True)
        # self.val_acc(predicted_labels, true_labels)
        # self.log("val_acc", self.val_acc, prog_bar=True)

    # def test_step(self, batch, batch_idx):
    #     loss, targets, preds = self._shared_step(batch)

    def configure_optimizers(self):
        optimizer = torch.optim.RMSprop(self.parameters(), lr=self.learning_rate)
        
        return optimizer



