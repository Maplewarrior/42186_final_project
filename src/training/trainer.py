from src.utils.misc import load_config
from tqdm import tqdm
import torch.optim as optim
import torch
import pdb
import torchvision
import matplotlib.pyplot as plt

def save_images(images, path, show=True, title=None, nrow=10):
    grid = torchvision.utils.make_grid(images, nrow=nrow)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    if title is not None:
        plt.title(title)
    plt.imshow(ndarr)
    plt.axis('off')
    if path is not None:
        plt.savefig(path, bbox_inches='tight', pad_inches=0)
    if show:
        plt.show()
    plt.close()

def denormalize(x):
    x = (x.clamp(-1, 1) + 1) / 2
    x = (x * 255).type(torch.uint8)
    return x


class Trainer:
    def __init__(self, model, train_loader, config_path: str, device: str) -> None:
        self.model = model
        self.train_loader = train_loader
        self.CFG = load_config(config_path)
        self.device=device
        self.__initialize_training_utils()
    
    def __initialize_training_utils(self):
        optimizer_name = self.CFG['training']['optimization']['optimizer_name']
        if optimizer_name == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), 
                                        lr=self.CFG['training']['optimization']['Adam']['lr'],
                                        betas=self.CFG['training']['optimization']['Adam']['betas'])
    def train(self):
        self.model.train()
        self.model.to(self.device)
        losses = []
        
        with tqdm(total=self.CFG['training']['n_epochs']*len(self.train_loader), desc="Training", unit="iter") as pbar:
            for epoch in range(self.CFG['training']['n_epochs']):
                for batch, _ in self.train_loader:
                    batch = batch.to(self.device)
                    loss = self.model.loss(batch)
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    losses.append(loss.item())
                    pbar.set_postfix({'Current loss': sum(losses)/len(losses)}, refresh=True)
                    pbar.update(1)
                
                if epoch % 50 == 0:
                    with torch.no_grad():
                        samples = self.model.sample(n_samples=20)
                    save_images(denormalize(samples), path = f'DDPM_samples/epoch{epoch+1}_samples.png', 
                                show=False, title=f'Epoch {epoch+1} samples')
                    
        return losses