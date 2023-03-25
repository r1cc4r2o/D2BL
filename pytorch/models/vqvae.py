import torch
import torch.nn as nn

""" [Basic implementation of Vector Quantised-Variational AutoEncoder]

    source paper vae: https://arxiv.org/abs/1312.6114v10
    source paper vqvae: https://arxiv.org/abs/1711.00937

    Useful resources:

        Blogs

            Understanding Variational Autoencoders (VAEs)
            + https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73
            Understanding Vector Quantized Variational Autoencoders (VQ-VAE)
            + https://shashank7-iitd.medium.com/understanding-vector-quantized-variational-autoencoders-vq-vae-323d710a888a

        YouTube

                VQ-GAN | Paper Explanation
                + https://www.youtube.com/watch?v=wcqLFDXaDO8
                VQ-VAEs: Neural Discrete Representation Learning
                + https://www.youtube.com/watch?v=VZFVUrYcig0
                AE, DAE, and VAE with PyTorch; Alfredo Canziani
                + https://www.youtube.com/watch?v=bZF4N8HR1cc

        Code
            Tutorial DeepMind VQGAN
            + https://github.com/deepmind/sonnet/blob/v1/sonnet/examples/vqvae_example.ipynb


"""