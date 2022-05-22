from .lenet import LeNet

def build_model(cfg):
    model = LeNet()
    return model
    