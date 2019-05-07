import torch
import torch.nn as nn


def mask_loss_l1(loss, params):
    return torch.sum(torch.abs(params)) + 1/loss

def mask_loss_l2(loss, params):
    return torch.sqrt(torch.sum(torch.pow(params, 2))) + 1/loss



