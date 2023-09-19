import torch
import torch.nn.functional as F



def copy_to_device(tensor: torch.Tensor, device):
    return tensor.to(device)
