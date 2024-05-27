import torch, pdb
import torch.nn as nn
import torch.distributions as td
from src.utils.misc import load_config

class DDPM(nn.Module):
    def __init__(self, unet, cfg: dict, device) -> None:
        super().__init__()
        self.name = 'DDPM'
        self.CFG = cfg
        self.unet = unet
        self.mse = nn.MSELoss(reduction='none')
        self.device = device
        self.T = self.CFG['DDPM']['T']
        self.p_uncond = self.CFG['DDPM']['p_uncond']
        self.__init_dependencies()
    
    def __init_dependencies(self):
        self.ts = torch.tensor(list(range(self.T-1, -1, -1)), dtype=torch.long).to(self.device)
        self.betas = self.unsqueeze(torch.linspace(1e-4, 0.02, self.T).to(self.device))
        self.sigmas = self.betas.to(self.device)
        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0).to(self.device)
        
    def forward(self, x: torch.tensor, y: torch.tensor = None):
        """
        Function that implements a single step of the training algorithm from the DDPM paper
        """
        # sample a time-step uniformly (one for each input in batch)
        t = torch.randint(low=0, high=self.T, size=(x.size(0),)).to(self.device)

        # sample noise from a standard multivariate Gaussian
        eps = td.Normal(loc=0, scale=1).sample(x.size()).to(self.device)
        # apply noise to the input image
        x_noised = x * torch.sqrt(self.alpha_bar[t]) + eps * torch.sqrt(1 - self.alpha_bar[t])
        
        if self.p_uncond > torch.rand(1).item(): # do uncoditional training with probability p_uncond
            y = None
        
        # do noise prediction
        predicted_noise = self.unet(x_noised, t, y)
        return predicted_noise, eps
    
    def unsqueeze(self, x: torch.tensor):
        return x[:, None, None, None]

    def loss(self, x, y):
        noise_pred, noise = self.forward(x, y)
        return self.mse(noise_pred, noise).mean()

    def sample(self, n_samples: int, y=None):
        """
        Function that implements the sampling algorithm from the DDPM paper
        """
        # draw x_T from a standard Gaussian
        x = td.Normal(loc=0, scale=1).sample((n_samples, self.CFG['data']['channels'], self.CFG['data']['H'], self.CFG['data']['W'])).to(self.device)
        
        for i in self.ts: # ts = [T-1, ..., 0]
        
            t = (torch.ones(n_samples, device=self.device) * i).long()
            if i > 0:
                z = td.Normal(loc=0, scale=1).sample(x.size()).to(self.device)
            else:
                z = torch.zeros_like(z)
  
            eps = self.unet(x, t, y)
            mu = (x - (1 - self.alphas[t]) / (torch.sqrt(1 - self.alpha_bar[t])) * eps) / torch.sqrt(self.alphas[t])
            noise = torch.sqrt(self.sigmas[t]) * z
            
            x = mu + noise 
            if i % 50 == 0:
                print(f'iter: {self.T-i}/{self.T}')
            
        
        return x
