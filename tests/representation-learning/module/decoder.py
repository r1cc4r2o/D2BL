
import torch
import torch.nn as nn


### TODO: the last layer of the encoder/decoder should be without activation function????



class Decoder(nn.Module):
    """Decoder for VQ-VAE
    
    Args:
        in_channels (int): Number of channels in the input image
        num_hiddens (int): Number of hidden channels
        num_decoder_layers (int): Number of decoder layers
        
    References:
    - [Oord et al., 2017](https://arxiv.org/abs/1711.00937)
    
    """
    def __init__(self, in_channels, num_hiddens, num_decoder_layers):
        super().__init__()

        self.num_decoder_layers = num_decoder_layers
        
        
        # Decoder
        self.decoder = nn.Sequential([
            nn.Sequential(
                
                nn.ConvTranspose2d(num_hiddens, num_hiddens // 2, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.ConvTranspose2d(num_hiddens // 2, num_hiddens // 4, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.ConvTranspose2d(num_hiddens // 2, num_hiddens // 4, kernel_size=3, stride=2, padding=1)
                
            ) for _ in range(num_decoder_layers)
            
        ])
            
            
    def forward(self, x):
        return self.decoder(x)