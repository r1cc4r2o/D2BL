import torch
import torch.nn as nn

### TODO: check if it works

""" [Basic implementation of the Transformer Encoder]

    Attention is all you need [Vaswani et. al., 2017]
    source : https://arxiv.org/abs/1706.03762

    Useful resources:

        Blogs

            The Illustrated Transformer
            + https://jalammar.github.io/illustrated-transformer/

        YouTube

                llustrated Guide to Transformers Neural Network: A step by step explanation
                + https://www.youtube.com/watch?v=4Bdc55j80l8
                Stanford CS224N: NLP with Deep Learning | Winter 2019 | Lecture 14 – Transformers and Self-Attention
                + https://www.youtube.com/watch?v=5vcj8kSwBCY
                10 Lesson - Self / cross, hard / soft attention and the Transformer
                + https://www.youtube.com/watch?v=fEVyfT-gLqQ

        Code
            Tutorial 6: Transformers and Multi-Head Attention (JAX)
            + https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html


"""


class TransformerEncoder(nn.Module):
    """Standard Encoder of the transformer architecture.
    
    Args:
        num_heads (int): Number of attention heads.
        feedforward_dim (int): Dimension of feedforward linear layer without activation function.
        embedding_dim (int): Dimension of embedding.
        num_transformer_encoder (int, optional): Number of transformer encoder. Defaults to 2.
        dropout (float, optional): Dropout rate. Defaults to 0.
        activation (str, optional): Activation function. Defaults to "relu".
        layer_norm_epsilon (float, optional): Epsilon value for layer normalization. Defaults to 1e-05.
        kernel_initializer (str, optional): Initializer for the kernel weights matrix (Conv2D, Dense, etc.). Defaults to "glorot_uniform".
        bias_initializer (str, optional): Initializer for the bias vector. Defaults to "zeros".
        name (str, optional): Name of the module. Defaults to "TransformerEncoder".
        
    References:
    - [Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)
    
    """
    
    
    def __init__(
                self, 
                num_heads,
                feedforward_dim,
                embedding_dim,
                num_transformer_encoder=2,
                dropout=0,
                activation=nn.GELU,
                layer_norm_epsilon=1e-05,
                kernel_initializer="glorot_uniform",
                bias_initializer="zeros",
                name="TransformerEncoder",
                **kwargs
            ):
        
        super().__init__(name=name, **kwargs)
        
        # number of heads multihead attention
        self.num_heads = num_heads
        
        # dimension of feedforward linear layer without activation function
        # inside the transformer encoder
        self.feedforward_dim = feedforward_dim
        
        # number of transformer encoder 
        # in the paper represented as L
        # in the image 
        # source : https://arxiv.org/abs/1706.03762
        self.num_transformer_encoder = num_transformer_encoder
        
        # dropout rate
        self.dropout = dropout
        
        # activation function
        self.activation = activation
        
        # epsilon value for layer normalization in the transformer encoder
        self.layer_norm_epsilon = layer_norm_epsilon
        
        # initializer for the kernel weights matrix (Conv2D, Dense, etc.)
        self.kernel_initializer = kernel_initializer
        
        # initializer for the bias vector
        self.bias_initializer = bias_initializer
        
        # multihead attention
        self.MHA = nn.MultiHeadAttention(embedding_dim, num_heads, dropout=dropout)
        
        # feedforward
        self.feedforward = nn.Sequential(
            nn.Linear(embedding_dim, feedforward_dim),
            activation(),
            nn.Linear(feedforward_dim, embedding_dim)
        )
        
        # Layer Normalization
        # Applies Layer Normalization over a mini-batch of inputs as described in the paper
        # [Layer Normalization](https://arxiv.org/abs/1607.06450)
        self.attention_layer_norm = nn.LayerNorm(embedding_dim, eps=layer_norm_epsilon)
        # I have initialized two normalization layers because we can 
        # play with the dimensionality of the latent representation
        self.feedforward_layer_norm = nn.LayerNorm(embedding_dim, eps=layer_norm_epsilon)
        
        
    def get_config(self):
        """Returns the config of the module."""
        config = super().get_config()
        config.update({
            "num_heads": self.num_heads,
            "feedforward_dim": self.feedforward_dim,
            "dropout": self.dropout,
            "activation": self.activation,
            "layer_norm_epsilon": self.layer_norm_epsilon,
            "kernel_initializer": self.kernel_initializer,
            "bias_initializer": self.bias_initializer,
        })
        return config
       
        
    def forward(self, input):
        """Forward pass of the transformer encoder.
        
        Args:
            input (torch.Tensor): Input tensor of shape (batch_size, channel, width, height).
            
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, channel, width, height).
            
        """
        
        # number of transformer encoder
        for _ in range(self.num_transformer_encoder):
            
            # multihead attention self attention
            attention_output = self.MHA(input, input, input)
            
            # add & norm
            attention_output = self.attention_layer_norm(input + attention_output)
            
            # feedforward
            feedforward_output = self.feedforward(attention_output)
            
            # add & norm
            feedforward_output = self.feedforward_layer_norm(attention_output + feedforward_output)
        
        return feedforward_output
        
        
        

        