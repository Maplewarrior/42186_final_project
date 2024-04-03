import torch
import torch.nn as nn
import torch.distributions as td
from torch.distributions.kl import kl_divergence as KL
import pdb

class GaussianEncoder(nn.Module):
    def __init__(self, H: int, W: int, D: int) -> None:
        super().__init__()
        self.H = H
        self.W = W
        self.D = D

        self.encoder_net = self.__get_encoder_net()
    def __get_encoder_net(self):
        encoder_net = nn.Sequential(
                                    nn.Flatten(), # flatten image [N x H*W*3]
                                    nn.Linear(self.H*self.W* 3, 784),
                                    nn.ReLU(),
                                    nn.Linear(784, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, 512),
                                    nn.ReLU(),
                                    nn.Linear(512, self.D*2)
                      )
        return encoder_net

    def forward(self, x: torch.tensor):
        mu, log_sigma = torch.split(self.encoder_net(x), self.D, dim=-1)
        return td.Independent(td.Normal(loc=mu, scale=torch.exp(log_sigma)), 1)

class ContinuousBernoulliDecoder(nn.Module):
    def __init__(self, H: int, W: int, D: int) -> None:
        super().__init__()
        self.H = H
        self.W = W
        self.D = D
        self.decoder_net = self.__get_decoder_net()
        self.sigmoid = nn.Sigmoid()

    def __get_decoder_net(self):
        decoder_net = nn.Sequential(nn.Linear(self.D, 512),
                      nn.ReLU(),
                      nn.Linear(512, 784),
                      nn.ReLU(),
                      nn.Linear(784, 784),
                      nn.RelU(),
                      nn.Linear(784, self.H * self.W * 3),
                      nn.Unflatten(-1), (3, self.H, self.W))

        return decoder_net
    
    def forward(self, z: torch.tensor):
        """
        Parameters:
            z: torch.tensor of size (N x D) N is the batch size and D denotes the latent dimension.
        """
        logits = self.decoder_net(z) # [N x 3 x H x W]
        return td.Independent(td.ContinuousBernoulli(probs=self.sigmoid(logits)), 3)

class VAE:
    def __init__(self, encoder, decoder, prior) -> None:
        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior

    def ELBO(self, x: torch.tensor):
        z = self.prior.sample()
        reconstruction = self.decoder(z).log_prob(x) # p(x | z)
        KL_term = KL(self.encoder(x), self.prior())
        return (reconstruction - KL_term).mean(dim=0) # average over batch dimension
    
    def sample(self, n_samples: int):
        z = self.prior.sample(torch.Size([n_samples]))
        return self.decoder(z).sample()
    
    def loss(self, x: torch.tensor):
        return -self.ELBO(x)

