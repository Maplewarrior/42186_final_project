import torch
import torch.nn as nn
from torchvision import datasets, transforms
from src.training.trainer import Trainer

from src.models.DDPM import DDPM
from src.models.unet import UNet

from src.utils.misc import load_config
from torchvision.utils import save_image

import pdb

if __name__ == '__main__':
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    ### MNIST data constants
    cfg = load_config('configs/config.yaml')
    cfg['H'] = 28
    cfg['W'] = 28
    cfg['C'] = 1

    unet = UNet(img_size=cfg['H'], c_in=cfg['C'], c_out=cfg['C'], device=device).to(device)
    model = DDPM(unet=unet, cfg=cfg, device=device).to(device)

    # Load a subset of MNIST and create data loaders
    def subsample(data, targets, num_data, num_classes):
        idx = targets < num_classes
        new_data = data[idx][:num_data].unsqueeze(1)
        new_targets = targets[idx][:num_data]
        return torch.utils.data.TensorDataset(new_data, new_targets)
    
    num_train_data = 4096
    num_test_data = 16  # we keep this number low to only compute a few geodesics
    num_classes = 3
    train_tensors = datasets.MNIST('data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
    
    train_data = subsample(train_tensors.data, train_tensors.targets, num_train_data, num_classes)
    mnist_train_loader = torch.utils.data.DataLoader(train_data, batch_size=32)
    
    trainer = Trainer(model, mnist_train_loader, 'configs/config.yaml', device=device)
    trainer.train()
    model.eval()
    with torch.no_grad():
        samples = model.sample(n_samples=64)
    
    save_image(samples, 'MNIST_samples.png')

    pdb.set_trace()
    print(f'Max: {samples.max()}\nMean: {samples.mean()}\nVar: {samples.var()}')
