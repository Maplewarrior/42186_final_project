import torch
import torchvision
from torchvision import transforms, datasets
from torchvision.utils import save_image
import matplotlib.pyplot as plt

import pdb

from src.models.DDPM import DDPM
from src.models.unet import UNet
from src.training.trainer import Trainer
from src.utils.misc import load_config



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

if __name__ == '__main__':
    #### SETUP MODEL
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    cfg = load_config('configs/config.yaml')
    unet = UNet(img_size=32, c_in=3, c_out=3, time_dim=128, device=device)
    model = DDPM(unet=unet,cfg=cfg, device=device)

    def subsample(dataloader, class_to_keep: int):
        keep_samples = []
        keep_labels = []

        for i, (batch, labels) in enumerate(dataloader):
            if type(class_to_keep) == int:
                mask = labels == class_to_keep
                if i == 0:
                    keep_samples = batch[mask, :, :, :]
                    keep_labels = labels[labels == class_to_keep]
                else:
                    keep_samples = torch.cat([keep_samples, batch[mask, :, :, :]], dim=0)
                    keep_labels = torch.cat([keep_labels, labels[labels == class_to_keep]])
                
        data = torch.utils.data.TensorDataset(keep_samples, keep_labels)
        return data

    #### Train
    # cifar_train = datasets.CIFAR10('data/', train=True, download=True,
    #                                 transform=transforms.Compose([
    #                                 transforms.ToTensor(), # from [0,255] to range [0.0,1.0]
    #                                 transforms.Normalize((0.5,), (0.5,))  # range [-1,1]
    #                                          ]))
    
    # cifar_train_loader = torch.utils.data.DataLoader(cifar_train, batch_size=cfg['training']['batch_size'])
    
    # model.train()
    
    # cifar_subsampled = subsample(cifar_train_loader, class_to_keep=0)
    # cifar_train_loader = torch.utils.data.DataLoader(cifar_subsampled, batch_size=cfg['training']['batch_size'])


    # trainer = Trainer(model, cifar_train_loader, 'configs/config.yaml', device)
    # losses = trainer.train()
    # torch.save(model.state_dict(), f='DDPM_cifar_weights.pt')

    sd = torch.load(f'DDPM_cifar_weights.pt', map_location=device)
    model.load_state_dict(sd)
    model = model.to(device)
    model.eval()
    with torch.no_grad():
        samples = model.sample(n_samples=64)
        # unnormalize pixel values
        # samples = (samples.clamp(-1, 1) + 1) / 2
        # samples *= 255

    save_images(denormalize(samples), path='CIFAR10_samples_denormalized.png', show=True, title='Class 0 samples')

    # pdb.set_trace()
    # print(f'Max: {samples.max()}\nMean: {samples.mean()}\nVar: {samples.var()}')
