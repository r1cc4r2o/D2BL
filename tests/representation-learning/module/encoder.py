import torch
import torch.nn as nn


### TODO: the last layer of the encoder/decoder should be without activation function????


class Encoder(nn.Module):
    """Encoder for VQ-VAE
    
    Args:
        in_channels (int): Number of channels in the input image
        num_hiddens (int): Number of hidden channels
        num_encoder_blokcs (int): Number of encoder blocks
        
    References:
    - [Oord et al., 2017](https://arxiv.org/abs/1711.00937)
    
    """
    
    def __init__(self, in_channels, num_hiddens, num_encoder_blokcs):
        super().__init__()
        
        # Encoder
        self.encoder = nn.Sequential([
            nn.Sequential(
                nn.Conv2d(in_channels, num_hiddens // 2, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(num_hiddens // 2, num_hiddens // 2, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(num_hiddens // 2, num_hiddens, kernel_size=3, stride=1, padding=1),
            ) for _ in range(num_encoder_blokcs)
        ])

            
            
            
