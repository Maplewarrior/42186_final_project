# Inspired by the AML course and https://github.com/Maplewarrior/02460_project_1/blob/main/src/models/priors.py
import torch
import torch.nn as nn
import torch.distributions as td
import pdb

class StdGaussianPrior(nn.Module):
    def __init__(self, D) -> None:
        super().__init__()
        self.D = D
        self.mu = nn.Parameter(torch.zeros((self.D)), requires_grad=False)
        self.sigma = nn.Parameter(torch.ones((self.D)), requires_grad=False)
    
    def forward(self):
        return td.Independent(td.Normal(loc=self.mu, scale=self.sigma), 1)

class MixtureOfGaussiansPrior(nn.Module):
    def __init__(self, num_components, latent_dim):
        super(MixtureOfGaussiansPrior, self).__init__()
        self.num_components = num_components
        self.latent_dim = latent_dim

        # Parameters for mixture weights (pi)
        self.mixture_weights = nn.Parameter(torch.randn(self.num_components), requires_grad=True)

        # Parameters for component means and standard deviations
        self.means = nn.Parameter(torch.randn(self.num_components, self.latent_dim), requires_grad=True)
        self.stds = nn.Parameter(torch.ones(self.num_components, self.latent_dim), requires_grad=True)

    def forward(self):
        # Use softmax to ensure the mixture weights sum to 1 and are non-negative
        mixture_weights = torch.nn.functional.softmax(self.mixture_weights, dim=0)

        # Create a categorical distribution for selecting the components
        categorical = td.Categorical(probs=mixture_weights)

        # Create a distribution for the components using Independent and Normal
        component_distribution = td.Independent(td.Normal(loc=self.means, scale=torch.exp(self.stds)), 1)

        # Combine the above into a MixtureSameFamily distribution
        prior = td.MixtureSameFamily(categorical, component_distribution)

        return prior
    
# Variational mixture of posteriors prior using pseudo-inputs
class VampPrior(nn.Module):
    def __init__(self, num_components, H, W, C, encoder):
        super(VampPrior, self).__init__()
        self.num_components = num_components
        self.H = H
        self.W = W
        self.C = C
        self.encoder = encoder

        # Parameters for mixture weights (pi) - learnable
        self.mixture_weights = nn.Parameter(torch.randn(self.num_components), requires_grad=True)
        # Parameters for pseudo inputs
        self.pseudo_inputs = nn.Parameter(torch.rand(self.num_components, self.C, self.H, self.W), requires_grad=True)

    def forward(self):  
        # Use softmax to ensure the mixture weights sum to 1 and are non-negative
        mixture_weights = torch.nn.functional.softmax(self.mixture_weights, dim=0)
        # Create a categorical distribution for selecting the components
        categorical = td.Categorical(probs=mixture_weights)
        
        component_distribution = self.encoder(self.pseudo_inputs)
        # Combine the above into a MixtureSameFamily distribution
        prior = td.MixtureSameFamily(categorical, component_distribution)
        return prior
    