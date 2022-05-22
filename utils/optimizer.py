import torch
import cfg

def build_optimizer(model):
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=cfg.initial_lr,
        momentum=0.5,
        weight_decay=0
    )
    return optimizer


