import torch, pdb
import torch.nn as nn
import torch.distributions as td
from src.utils.misc import load_config
from src.models.utils import Reshape
class DDPM(nn.Module):
    def __init__(self, unet, cfg: dict) -> None:
        super().__init__()
        self.CFG = cfg
        self.unet = unet
        self.mse = nn.MSELoss()
        self.device = ('cuda' if torch.cuda.is_available() else 'cpu')
        self.T = self.CFG['DDPM']['T']
        self.__init_dependencies()
    
    def __init_dependencies(self):
        
        # self.ts = torch.linspace(start=0, stop=T, step=(1,), dtype=torch.int)
        self.ts = torch.tensor(list(reversed(range(self.T))), dtype=torch.int32).to(self.device)
        self.betas = torch.linspace(1e-4, 0.02, self.T).to(self.device)
        self.sigmas = self.betas.to(self.device)
        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=-1)
        
    def forward(self, x: torch.tensor):
        """
        Function that ....
            Parameters:
            - x: torch.tensor of dimensions [N x 3 x H x W]
        """
        # sample a time-step uniformly (one for each input in batch)
        t = torch.randint(low=0, high=self.T, size=(x.size(0),))
        # sample noise from a standard multivariate Gaussian
        eps = td.Normal(loc=0, scale=1).sample((x.size(0),)).to(self.device)
        eps = self.unsqueeze(eps)
        # compute a prediction from the unet
        pred = self.unet(self.unsqueeze(torch.sqrt(self.alpha_bar[t])) * x + self.unsqueeze(torch.sqrt(1 - self.alpha_bar[t])) * eps, t)
        return eps, pred
    
    def unsqueeze(self, x: torch.tensor):
        return x[:, None, None, None]

    def loss(self, x):
        eps, pred = self.forward(x)
        return self.mse(eps, pred)

    def sample(self, n_samples: int):

        # draw x_T from a standard Gaussian
        x = td.Normal(loc=0, scale=1).sample((n_samples,self.CFG['data']['channels'], self.CFG['data']['H'], self.CFG['data']['W'])).to(self.device)
        for t in self.ts:
            
            if t > 1:
                z = td.Normal(loc=0, scale=1).sample((n_samples,)).to(self.device)
            else:
                z = torch.zeros_like(z)
            z = self.unsqueeze(z)
           
            x = 1 / torch.sqrt(self.alphas[t]) * (x - ((1-self.alphas[t])/ (torch.sqrt(1-self.alpha_bar[t])) * self.unet(x, t))) + self.sigmas[t]*z
            if t % 2 == 0:
                print(f'iter: {self.T-t}/{self.T}')
        return x
