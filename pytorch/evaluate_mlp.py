import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from models.mlp_wd import MLPWD
from models.mlp import MLP


""" [EVALUATION MLP / MLPWD]
    
    Datasets
        + MNIST (classification)
        + CIFAR10 (classification)
    
    Evaluation metrics
        + Accuracy
        + Misclassification error
        + Average F-score 
        + Heatmap confusion matrix
        

"""