from src.utils.misc import load_config
from tqdm import tqdm
import torch.optim as optim

class Trainer:
    def __init__(self, model, train_loader, config_path: str) -> None:
        self.model = model
        self.train_loader = train_loader
        self.CFG = load_config(config_path)
        self.__initialize_training_utils()
    
    def __initialize_training_utils(self):
        optimizer_name = self.CFG['training']['optimization']['optimizer_name']
        if optimizer_name == 'Adam':
            self.optimizer = optim.Adam(self.model.parameters(), 
                                        lr=self.CFG['training']['optimization']['Adam']['lr'],
                                        betas=self.CFG['training']['optimization']['Adam']['betas'])
            
    def train(self):
        self.model.train()
        losses = []
        with tqdm(total=self.n_iter, desc="Training", unit="iter") as pbar:
            for batch, _ in self.train_loader():
                self.optimizer.zero_grad()
                loss = self.model.loss(batch)
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                pbar.set_postfix({'Current loss': loss.item()}, refresh=True)
                pbar.update(1)
                    
        return losses
