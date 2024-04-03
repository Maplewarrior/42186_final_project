import torch
import torch.nn as nn
import torch.distributions as td

class StdGaussianPrior(nn.Module):
    def __init__(self, D) -> None:
        self.D = D
        self.mu = nn.Parameter(torch.zeros((self.D)), requires_grad=False)
        self.sigma = nn.Parameter(torch.ones((self.D)), requires_grad=False)
    
    def forward(self):
        return td.Independent(td.Normal(loc=self.mu, scale=self.sigma), 1)
