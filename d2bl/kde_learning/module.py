import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader

import numpy as np
from sklearn.neighbors import KernelDensity

##############################################################
n_neurons = 64



###############################################################

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_loss /= len(train_loader.dataset)
    accuracy = 100. * correct / total
    return train_loss, accuracy

###############################################################

def evaluate(device, model, dataloader):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

###############################################################

def extract_activations_layers(layers):
    """ Extract for each layer the activations

    Args:
        layers (np.array): shape (layer_activation, batch_size, number_of_neurons)

    Returns:
        np.array: shape (layer_activation, batch_size, number_of_activations)
    """

    return np.array([np.array([np.array(h) for h in l]) for l in layers])

###############################################################

def extract_activations_per_sample(layers, mask = False):
    """ Extract for each sample the activations 
    for each layer and store them in a list.

    Args:
        layers (np.array): shape (layer_activation, batch_size, number_of_neurons)

    Returns:
        np.array: shape (batch_size, number_of_activations)
    """

    if mask == True:
        # mask the activations to remove zeros
        mask = layers != 0
        layers = [[np.array(h[m]) for h, m in zip(l,sm)] 
                for l, sm in zip(layers, mask)]
        
    return np.array([layers[:,i,:].flatten().reshape(-1, 1) for i in range(layers.shape[1])])

###############################################################

def get_sampled_activations(activations, bandwidth = 0.2):
    """ Sample the activations using KDE

    Args:
        activations (np.array): shape (batch_size, number_of_activations)

    Returns:
        np.array: shape (batch_size, number_of_activations)
    """

    return torch.from_numpy(np.array([KernelDensity(kernel="gaussian", bandwidth=bandwidth).fit(a).sample([n_neurons]) for a in activations], dtype="float32")).squeeze(2)

###############################################################

def wd(layers: list()):
    """ Compute the weight decay for each layer

    Args:
        layers (list): list of layers

    Returns:
        torch.tensor: weight decay

    """
    return get_sampled_activations(
                list(
                    extract_activations_per_sample(
                            extract_activations_layers(layers), 
                            mask=False
                        )
                ), 
                bandwidth=0.2
            )

###############################################################
