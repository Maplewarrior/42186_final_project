import torch
import torch.nn as nn
from src.utils.misc import load_config

class DDPM(nn.Module):
    def __init__(self, unet, config_path) -> None:
        super().__init__()
        self.CFG = load_config(config_path)
        self.unet = unet
        self.__init_dependencies()
    
    def __init_dependencies(self):
        T = self.CFG['DDPM']['T']
        self.ts = torch.linspace(start=0, stop=T, step=(1,), dtype=torch.int)
        self.betas = torch.linspace(1e-4, 0.02, (T,))
        self.sigmas = self.betas
        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas)
    

    def forward(self):
        pass
    
    def loss(self):
        pass

    def sample(self):
        pass