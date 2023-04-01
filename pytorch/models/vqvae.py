import torch
import torch.nn as nn


### TODO: Implement VQ-VAE
### TODO: fix the number of cluster in the ema case 
#           'n = self.ema_cluster_size.sum()' line 121

# Set seed for reproducibility
torch.manual_seed(0)

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


class QuantizedCodebook(nn.Module):
    """ Quantized codebook for VQ-VAE
    """
    def __init__(self, num_embeddings, embedding_dim, beta=0.25, ema=False, _lambda=0.99):
        super().__init__()

        # embedding dimension
        self._embedding_dim = embedding_dim

        # number of embeddings
        self.num_embeddings = num_embeddings
        
        # Beta make the algorithm more robust
        # make sure that the output does not
        # grow too much.
        # We can chose beta in between 0.1 and 
        # 2.0, the authors of the paper made all
        # the experiments with beta = 0.25.
        # source: https://arxiv.org/abs/1711.00937
        self.beta = beta

        # Create embedding table
        self.embedd = nn.Embedding(num_embeddings, embedding_dim)

        # random uniform initialization of the embedding table 
        # between -1 / num_embeddings and 1 / num_embeddings
        self.embedd.w.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

        # https://arxiv.org/pdf/1805.11063.pdf
        # source: https://arxiv.org/pdf/1803.03382.pdf
        self.ema = ema # exponential moving average
        # In their paper they use lambda = 0.99
        # for all the experiments.
        # source: https://arxiv.org/pdf/1803.03382.pdf
        self._lambda = _lambda
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_w", self.embedd.weight.data)



    def forward(self, x):

        x = x.reshape(-1, self._embedding_dim)

        # Compute the distances
        # for each encoded vector here we produce a table of distances
        # with respect to all the codebook vectors
            # x.t() transpose the dimensions 0 and 1
            #   source: https://pytorch.org/docs/stable/generated/torch.t.html
        distances = (x.pow(2).sum(1, keepdim=True) - 2 * x @ self.embedd.w.t() + self.embedd.w.pow(2).sum(1, keepdim=True).t())

        # Encoding
        # find the closest codebook vector
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and unflatten
        quantized = torch.matmul(encodings, self.embedd.w).view(x.shape)

        # Loss

        # Third term of the loss freezing the codebook vectors
        # source: https://arxiv.org/abs/1711.00937
        e_latent_loss = (x - quantized.detach()).pow(2).mean()

        # Second term of the loss freezing the input
        # source: https://arxiv.org/abs/1711.00937
        q_latent_loss = (x.detach() - quantized).pow(2).mean()

        # Compuding the loss
        # Equation (3) paper: https://arxiv.org/abs/1711.00937
        loss = q_latent_loss + self.beta * e_latent_loss


        # source: https://arxiv.org/pdf/1803.03382.pdf
        if self.ema:
            avg_probs = torch.mean(encodings, 0)

            # https://arxiv.org/pdf/1803.03382.pdf Equation (9)
            self.ema_cluster_size = (self.ema_cluster_size * self._lambda + (1 - self._lambda) * avg_probs)

            # https://arxiv.org/pdf/1803.03382.pdf Equation (10)
            n = self.ema_cluster_size.sum() # in the paper they write ... 'the embedding e_j being subsequently updated'
            self.ema_w = (self.ema_w * self._lambda + (1 - self._lambda) * self.embedd.w * n)
            cluster_size = self.ema_cluster_size + 1e-10
            self.embedd.w.data = self.ema_w / cluster_size.unsqueeze(1)


        # When ever we compute backpropagation we exclude 
        # the gradient of the quantized vector with respect 
        # to the input from the gradient computation to
        # update the weights of the encoder.
        # source: https://arxiv.org/abs/1711.00937 (page 4, sentence 2)
        # visualization image: https://miro.medium.com/v2/resize:fit:828/format:webp/1*pTECMRydvo-e3j58W75Hnw.png
        quantized = x + (quantized - x).detach()


        avg_probs = torch.mean(encodings, 0).detach()
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return loss, quantized, perplexity, encodings