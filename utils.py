import torch

def wd(layers: list()):
    return 1 - len(layers)/10