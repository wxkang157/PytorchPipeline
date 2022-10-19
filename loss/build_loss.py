import torch
from .tripletloss import TripletLoss


def get_loss(cfg):
    if cfg.CRITERION.NAME == 'CE':
        loss_function = torch.nn.CrossEntropyLoss()
    else:
        loss_function = TripletLoss(t1=0, t2=0, beta=2)
    return loss_function