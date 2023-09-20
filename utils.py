import torch
import torch.nn.functional as F



def copy_to_device(tensor: torch.Tensor, device=torch.device("cuda")):
    return tensor.to(device)
