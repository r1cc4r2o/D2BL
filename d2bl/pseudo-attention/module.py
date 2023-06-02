import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision.datasets as datasets
import torchvision.transforms as transforms

from torch.utils.data import DataLoader


##############################################################
### GLOBAL VARIABLES
CROSSOVER_MAGNITUDE = 0.3
MUTATION_FACTOR = 0.3


##############################################################


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


##############################################################


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


##############################################################


def head_batched_attention_mechanism(Q, K, V):
    """
    Args:
        Q: (batch_size, num_heads, num_layer, activation_size)
        K: (batch_size, num_heads, num_layer, activation_size)
        V: (batch_size, num_heads, activation_size, 1) # activations in the current layer

    Returns:
        attention: (batch_size, num_heads, activation_size)

        attention mechanism
        (batch_size, num_heads, activation_size, activation_size)
        attention = torch.matmul(Q, K.transpose(-1,-2))
        attention = attention / torch.sqrt(torch.tensor(activation_size).float())

        (batch_size, num_heads, activation_size, 1)
        attention = nn.Softmax(dim=-1)(attention)

        (batch_size, num_heads, activation_size, 1)
        attention = torch.matmul(attention, V)

        (batch_size, num_heads, activation_size)
        attention = attention.squeeze(-1)

    """

    return (nn.Softmax(dim=-1)(
                torch.matmul( 
                            Q.transpose(-1,-2) , 
                            K
                )/torch.sqrt(torch.tensor(V.shape[2]))    
            ) @ V).squeeze(-1)


##############################################################


def get_activations_per_object(activations):
    """ Get the activations for each object per layer

    Args:
        activations (torch.Tensor): shape (num_layers, batch_size, number_activations)

    Returns:
        torch.Tensor: shape (nr_object, num_layers, activation_for_each_layer)

    """
    return torch.stack([activations[:,i,:] for i in range(activations.shape[1])])


##############################################################


def get_layer_activations(activations):
    """ Get the activations for each layer for each sample

    Args:
        activations (torch.Tensor): shape (batch_size, number_activations)
        batch_size (int): batch size
        number_activations (int): number of activations

    Returns:
        torch.Tensor: shape (nr_object, num_layers, activation_for_each_layer)

    """
    return get_activations_per_object(torch.stack(activations))


################################################################

def attention_map_crossover(attention_map):
    """ Apply the crossover over the attention maps of each head.
    The crosseover consists in picking a random index in the maxtrix over
    the columns and swapping the values in between the columns of the
    attention map.
    
    Args:
        attention_map (torch.Tensor): shape (batch_size, number_of_heads, activation_size, activation_size)
        
    Returns:
        torch.Tensor: shape (batch_size, number_of_heads, activation_size, activation_size)
    
    """
    
    # get the crossover magnitude
    crossover_magnitude = CROSSOVER_MAGNITUDE
    
    # get the batch size
    dim_batch = attention_map.shape[0]
    
    # get the number of heads
    number_of_heads = attention_map.shape[1]
    
    for idx_batch in range(dim_batch):
        for idx_head in range(number_of_heads):
            
            # get the crossover index
            crossover_index = attention_map.shape[2] - int(attention_map.shape[2]*crossover_magnitude)
            
            # get two random indexes
            random_index_1 = torch.randint(0, attention_map.shape[2],(1,))[0]
            random_index_2 = torch.randint(0, attention_map.shape[2],(1,))[0]
            
            # swap the values in that position over the columns
            for idx, (x_1, x_2) in enumerate(zip(attention_map[idx_batch][idx_head][random_index_1][crossover_index:].detach(), attention_map[idx_batch][idx_head][random_index_2][crossover_index:].detach())):
                
                # debug
                # print(attention_map[idx_batch][idx_head].shape, random_index_1, random_index_2, crossover_index)         
                # print(x_1, x_2,idx_batch, idx_head, idx)
                
                # swap the values in that position over the columns
                attention_map[idx_batch][idx_head][random_index_1][crossover_index+idx] = x_2 # make crossover
                attention_map[idx_batch][idx_head][random_index_2][crossover_index+idx] = x_1 # make crossover
    
    return attention_map

################################################################

def mutate_attention_map(attention_map):
    """ Mutate the attention map by making an elementwise multiplication with 
    a random tensor with values between 1-mutation_factor and 1+mutation_factor
    
    Args:
        attention_map (torch.Tensor): shape (batch_size, num_heads, activation_size, activation_size)
        
    Returns:    
        torch.Tensor: shape (batch_size, num_heads, activation_size, activation_size)
    
    
    """
    # get the mutation factor
    mutation_factor = MUTATION_FACTOR
    # return the mutated attention map
    # multiplied elementwise with a 
    # random matrix with values between 
    # 1-mutation_factor and 1+mutation_factor
    return torch.mul(attention_map, torch.randn(attention_map.shape).uniform_(1-mutation_factor,1+mutation_factor).to(attention_map.device))
    
    
################################################################


def head_batched_attention_mechanism(Q, K, V):
    """
    Args:
        Q: (batch_size, num_heads, num_layer, activation_size)
        K: (batch_size, num_heads, num_layer, activation_size)
        V: (batch_size, num_heads, activation_size, 1) # activations in the current layer

    Returns:
        attention: (batch_size, num_heads, activation_size)

        # attention mechanism
        # # (batch_size, num_heads, activation_size, activation_size)
        # attention = torch.matmul(Q, K.transpose(-1,-2))
        # attention = attention / torch.sqrt(torch.tensor(activation_size).float())

        # # (batch_size, num_heads, activation_size, 1)
        # attention = nn.Softmax(dim=-1)(attention)

        # # (batch_size, num_heads, activation_size, 1)
        # attention = torch.matmul(attention, V)

        # # (batch_size, num_heads, activation_size)
        # attention = attention.squeeze(-1)

    """
    
    # with probability p
    p = torch.rand(1)
    
    # p <= 0.6 apply the mutation only
    if p <= 0.6:
        return (nn.Softmax(dim=-1)(
                    mutate_attention_map(torch.matmul( 
                                Q.transpose(-1,-2) , 
                                K
                    )/torch.sqrt(torch.tensor(8)))    
                ) @ V).squeeze(-1)
        
    # p > 0.6 apply the crossover only
    else:
        return (nn.Softmax(dim=-1)(
                attention_map_crossover(torch.matmul( 
                            Q.transpose(-1,-2) , 
                            K
                )/torch.sqrt(torch.tensor(8)))    
            ) @ V).squeeze(-1)


####################################################################


def get_activations_per_object(activations):
    """ Get the activations for each object per layer

    Args:
        activations (torch.Tensor): shape (num_layers, batch_size, number_activations)

    Returns:
        torch.Tensor: shape (nr_object, num_layers, activation_for_each_layer)

    """
    return torch.stack([activations[:,i,:] for i in range(activations.shape[1])])


####################################################################


def get_layer_activations(activations):
    """ Get the activations for each layer for each sample

    Args:
        activations (torch.Tensor): shape (batch_size, number_activations)
        batch_size (int): batch size
        number_activations (int): number of activations

    Returns:
        torch.Tensor: shape (nr_object, num_layers, activation_for_each_layer)

    """
    return get_activations_per_object(torch.stack(activations))



####################################################################


class LinW_Attention_Module(nn.Module):
    def __init__(self, dim_emb, n_head) -> None:
        super(LinW_Attention_Module, self).__init__()

        assert dim_emb % n_head == 0, 'dim_emb must be divisible by n_head'

        self.dim_emb = dim_emb
        self.n_head = n_head

        self.W_Q = nn.Linear(dim_emb, dim_emb)
        self.W_K = nn.Linear(dim_emb, dim_emb)
        self.W_V = nn.Linear(dim_emb, dim_emb)

        self.W_O = nn.Linear(dim_emb*n_head, dim_emb)

    def forward(self, Q, K, V):
        # get the shape of the input
        batch_size, activation_size, activation_size = Q.size()
        
        # check the device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # move to device
        Q.to(device)
        K.to(device)
        V.to(device)
        
        # reshape Q, K, V
        # parallelize over the number of heads
        # (batch_size, num_heads, num_layer, activation_size)
        Q = torch.stack([Q for _ in range(self.n_head)], 1)
        K = torch.stack([K for _ in range(self.n_head)], 1)
        V = torch.stack([V for _ in range(self.n_head)], 1)

        # apply linear transformation
        Q = self.W_Q(Q)
        K = self.W_K(K)
        V = self.W_V(V).reshape(batch_size, self.n_head, activation_size, 1)
        
        # apply attention mechanism
        out_attention = head_batched_attention_mechanism(Q, K, V).reshape(batch_size, self.n_head*activation_size)

        # apply linear transformation
        return self.W_O(out_attention)
    

####################################################################


class LinW_Attention_Module_C_M(nn.Module):
    def __init__(self, dim_emb, n_head) -> None:
        super(LinW_Attention_Module_C_M, self).__init__()

        assert dim_emb % n_head == 0, 'dim_emb must be divisible by n_head'

        self.dim_emb = dim_emb
        self.n_head = n_head

        self.W_O = nn.Linear(dim_emb*n_head, dim_emb)

    def forward(self, Q, K, V):
        # get the shape of the input
        batch_size, activation_size, activation_size = Q.size()
        
        # check the device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # move to device
        Q.to(device)
        K.to(device)
        V.to(device)
        
        # reshape Q, K, V
        # parallelize over the number of heads
        # (batch_size, num_heads, num_layer, activation_size)
        Q = torch.stack([Q for _ in range(self.n_head)], 1)
        K = torch.stack([K for _ in range(self.n_head)], 1)
        V = torch.stack([V for _ in range(self.n_head)], 1)

        V = V.reshape(batch_size, self.n_head, activation_size, 1)
        
        # apply attention mechanism
        out_attention = head_batched_attention_mechanism(Q, K, V).reshape(batch_size, self.n_head*activation_size)

        # apply linear transformation
        return self.W_O(out_attention)
    



