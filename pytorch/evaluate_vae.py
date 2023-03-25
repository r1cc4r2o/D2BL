import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader




""" [EVALUATION VAE / VAEWD]
    
    Datasets
        + MNIST 
        + Fashion-MNIST 
    
    Evaluation metrics
        + Mean reconstruction error
        + Latent space clusters representation
        + Heatmap confusion matrix reconstruction 
            error between the classes
        

"""