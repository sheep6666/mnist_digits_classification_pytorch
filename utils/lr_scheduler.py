import torch

import cfg

def build_scheduler(optimizer):
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=cfg.step_size, 
        gamma=cfg.gamma, 
        last_epoch=-1, 
        verbose=False
    )
    return scheduler
    
