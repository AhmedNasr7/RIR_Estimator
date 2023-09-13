import lightning as L
import torch
import torch.nn.functional as F

class PeriodicLRDecayCallback(L.Callback):
    def __init__(self, decay_epochs, decay_factor):
        self.decay_epochs = decay_epochs
        self.decay_factor = decay_factor

    def on_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.decay_epochs == 0:
            for param_group in trainer.optimizers[0].param_groups:
                param_group['lr'] *= self.decay_factor