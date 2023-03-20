import torch

def wd(layers: list()):
    return 1 - len(layers)/10 if len(layers)>0 else 1