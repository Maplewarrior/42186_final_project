import torch
import torch.nn as nn
import torch.distributions as td
from torch.distributions.kl import kl_divergence as KL
from src.models.utils import Reshape
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
                                    nn.Linear(self.H*self.W* 3, 4800),
                                    nn.ReLU(),
                                    nn.Linear(4800, 2048),
                                    nn.ReLU(),
                                    nn.Linear(2048, 2048),
                                    nn.ReLU(),
                                    nn.Linear(2048, self.D*2))
        encoder_net_cnn = nn.Sequential(
                                        nn.Conv2d(3, 8, kernel_size=4, stride=2, padding=1),
                                        nn.Softplus(),
                                        nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=1),
                                        nn.Softplus(),
                                        nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
                                        nn.Softplus(),
                                        nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=0),
                                        nn.Softplus(),
                                        nn.AdaptiveAvgPool2d(output_size=1),
                                        Reshape((64,)),
                                        nn.Linear(64, 128),
                                        nn.Linear(128, self.D*2)
                                    )

        return encoder_net_cnn

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
        decoder_net = nn.Sequential(
                                    nn.Linear(self.D, 1024),
                                    nn.ReLU(),
                                    nn.Linear(1024, 2048),
                                    nn.ReLU(),
                                    nn.Linear(2048, 2048),
                                    nn.ReLU(),
                                    nn.Linear(2048, self.H * self.W * 3),
                                    nn.Unflatten(-1, (3, self.H, self.W)))
        
        decoder_net_cnn = nn.Sequential(
                                        nn.Linear(self.D, 1024),
                                        Reshape((1024, 1, 1)),
                                        nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=0),
                                        nn.Softplus(),
                                        nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4,stride=2, padding=1),
                                        nn.Softplus(),
                                        nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4,stride=2, padding=1),
                                        nn.Softplus(),
                                        nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4,stride=2, padding=2),
                                        nn.Softplus(),
                                        nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=11, stride=1, padding=0),
                                        # nn.ReLU(),
                                        # nn.ConvTranspose2d(in_channels=48, out_channels=1, kernel_size=4,stride=2, padding=1),
                                        nn.Sigmoid()
                                    ) # (40-1) * s + k - 2*p
        return decoder_net_cnn
    
    def forward(self, z: torch.tensor):
        """
        Parameters:
            z: torch.tensor of size (N x D) N is the batch size and D denotes the latent dimension.
        """
        logits = self.decoder_net(z) # [N x 3 x H x W]
        return td.Independent(td.ContinuousBernoulli(probs=self.sigmoid(logits)), 3)

class MultivariateGaussianDecoder(nn.Module):
    def __init__(self, H, W, D, learn_variance=True):
        """
        Define a multivariate gaussian decoder distribution based on a given decoder network.

        Parameters: 
        encoder_net: [torch.nn.Module]             
           The decoder network that takes as a tensor of dim `(batch_size, M) as
           input, where M is the dimension of the latent space, and outputs a
           tensor of dimension (batch_size, feature_dim1, feature_dim2).
        """
        super().__init__()
        self.D = D
        self.H = H
        self.W = W

        self.decoder_net = self.__get_decoder_net()
        self.learn_variance = learn_variance
        self.log_var = nn.Parameter(torch.log(torch.ones(self.H, self.W) * .5), requires_grad=learn_variance)
    
    def __get_decoder_net(self):
        # decoder_net = nn.Sequential(
        #                             nn.Linear(self.D, 1024),
        #                             nn.ReLU(),
        #                             nn.Linear(1024, 2048),
        #                             nn.ReLU(),
        #                             nn.Linear(2048, 2048),
        #                             nn.ReLU(),
        #                             nn.Linear(2048, self.H * self.W * 3),
        #                             nn.Unflatten(-1, (3, self.H, self.W)))
        
        decoder_net_cnn = nn.Sequential(
                                        nn.Linear(self.D, 1024),
                                        Reshape((1024, 1, 1)),
                                        nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=4, stride=2, padding=0),
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4,stride=2, padding=1),
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4,stride=2, padding=1),
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4,stride=2, padding=2),
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=11, stride=1, padding=0),
                                        nn.Sigmoid()
                                        )
                                        # nn.ReLU(),
                                        # nn.ConvTranspose2d(in_channels=48, out_channels=1, kernel_size=4,stride=2, padding=1),
                                        
        #                             ) # (40-1) * s + k - 2*p
        # decoder_net_cnn = nn.Sequential(nn.Linear(self.D, 256 * (self.H // 8) * (self.H // 8)),
        #                                 nn.ReLU(),
        #                                 nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
        #                                 nn.ReLU(),
        #                                 nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
        #                                 nn.ReLU(),
        #                                 nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
        #                                 nn.Sigmoid()
                                        
        # )
        return decoder_net_cnn

    def forward(self, z):
        """
        Given a batch of latent variables, return a Gaussian distribution over the data space.

        Parameters:
        z: [torch.Tensor] 
           A tensor of dimension `(batch_size, M)`, where `M` is the dimension of the latent space.
        """
        # pdb.set_trace()
        dec_out = self.decoder_net(z)
        std = torch.exp(.5 * self.log_var)
        return td.Independent(td.Normal(loc=dec_out, scale=std), 3)

class VAE(nn.Module):
    def __init__(self, encoder, decoder, prior) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.prior = prior

    def ELBO(self, x: torch.tensor):
        q = self.encoder(x)
        z = q.rsample() # reparameterization trick
        reconstruction = self.decoder(z).log_prob(x) # ln p(x|z)
        # KL_term = q.log_prob(z) - self.prior().log_prob(z)
        KL_term = KL(q, self.prior())
        return (reconstruction - KL_term).mean(dim=0) # average over batch dimension
    
    def sample(self, n_samples: int):
        z = self.prior().sample(torch.Size([n_samples]))
        return self.decoder(z).sample()
    
    def loss(self, x: torch.tensor):
        return -self.ELBO(x)

