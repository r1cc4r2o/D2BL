import torch
import torch.nn as nn

""" [Basic implementation of Vector Quantised-Variational AutoEncoder]

    source paper vae: https://arxiv.org/abs/1312.6114v10
    source paper vqvae: https://arxiv.org/abs/1711.00937

    Useful resources:

        YouTube
            VQ-GAN | Paper Explanation
            + https://www.youtube.com/watch?v=wcqLFDXaDO8
            VQ-VAEs: Neural Discrete Representation Learning
            + https://www.youtube.com/watch?v=VZFVUrYcig0
            AE, DAE, and VAE with PyTorch; Alfredo Canziani
            + https://www.youtube.com/watch?v=bZF4N8HR1cc




"""