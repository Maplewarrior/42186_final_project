from src.utils.misc import load_config
from tqdm import tqdm
import torch.optim as optim
import torch
import pdb
import torchvision
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import os
import wandb

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
    def __init__(self, model, train_loader, config_path: str, device: str, uuid: str, wandb: wandb = None) -> None:
        self.device=device
        self.CFG = load_config(config_path)
        self.n_classes = self.CFG['data']['n_classes']
        self.row_idxs = list(range(self.CFG['training']['batch_size']))
        self.model = model
        self.train_loader = train_loader
    
        self.__initialize_training_utils()
        self.wandb = wandb
        self.uuid = uuid
    
    def __initialize_training_utils(self):
        optimizer_name = self.CFG['training']['optimization']['optimizer_name']
        if optimizer_name == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), 
                                        lr=self.CFG['training']['optimization']['Adam']['lr'],
                                        betas=self.CFG['training']['optimization']['Adam']['betas'])
    
    def to_onehot(self, y):
        y_onehot = torch.zeros((y.size(0), self.n_classes), device=self.device)
        y_onehot[self.row_idxs, y] = 1
        return y_onehot

    def train(self):
        self.model.train()
        self.model.to(self.device)
        losses = []
        
        with tqdm(total=self.CFG['training']['n_epochs']*len(self.train_loader), desc="Training", unit="iter") as pbar:
            for epoch in range(self.CFG['training']['n_epochs']):
                for batch, label in self.train_loader:
                    batch = batch.to(self.device)
                    label = label.to(self.device)

                    loss = self.model.loss(batch, self.to_onehot(label))
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    losses.append(loss.item())
                    if self.wandb: wandb.log({"epoch": epoch, "loss": sum(losses)/len(losses)})
                    pbar.set_postfix({'Current loss': sum(losses)/len(losses)}, refresh=True)
                    pbar.update(1)
                
                if (epoch+1) % 50 == 0:
                    with torch.no_grad():
                        samples = self.model.sample(n_samples=20)
                    os.makedirs(f"DDPM_samples/{self.uuid}/", exist_ok=True)
                    image_path = f'DDPM_samples/{self.uuid}/epoch{epoch+1}_samples.png'
                    if self.model.name == 'DDPM':
                        save_images(denormalize(samples), path = image_path, 
                                    show=False, title=f'Epoch {epoch+1} samples')
                    
                    elif self.model.name == 'VAE':
                        os.makedirs(f"VAE_samples/{self.uuid}/", exist_ok=True)
                        image_path = f'VAE_samples/{self.uuid}/epoch{epoch+1}_samples.png'
                        save_image(samples, fp=image_path)

                    if self.wandb: wandb.log({"sample_images": wandb.Image(image_path, caption=f'Epoch {epoch+1} samples')})
                    
                    checkpoint = {'model': self.model.state_dict(),
                                  'optimizer': self.optimizer.state_dict(),
                                  'config': self.CFG}

                    checkpoint_path = f'checkpoints/{self.uuid}/checkpoint_{epoch}epochs.pt'
                    os.makedirs(f"checkpoints/{self.uuid}/", exist_ok=True)
                    torch.save(checkpoint, checkpoint_path)

        return losses