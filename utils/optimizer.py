import torch

def build_optimizer(model):
    optimizer = torch.optim.SGD(
        model.parameters(), 
        lr=1e-1,
        momentum=0.5,
        weight_decay=0
    )
    return optimizer


