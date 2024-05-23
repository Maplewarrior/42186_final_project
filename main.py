import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
from torchvision.utils import save_image
import matplotlib.pyplot as plt

from src.data_utils.dataloader import PokemonDataset
from src.training.trainer import Trainer
from src.models.DDPM import DDPM
from src.models.unet import UNet
from src.models.VAE import VAE, ContinuousBernoulliDecoder, GaussianEncoder, MultivariateGaussianDecoder
from src.models.vae_priors import StdGaussianPrior
from src.utils.misc import load_config

def build_model(model_type: str, CFG: dict, device: str):
    H = CFG['data']['H']
    W = CFG['data']['W']
    C = CFG['data']['channels']
    D = CFG['VAE']['D']

    if model_type == 'VAE':        
        prior = StdGaussianPrior(D)
        encoder = GaussianEncoder(H, W, D)
        decoder = MultivariateGaussianDecoder(H, W, D)
        model = VAE(encoder, decoder, prior).to(device)
    
    elif model_type == 'DDPM':
        unet = UNet(img_size=H, c_in=C, c_out=C, device=device).to(device)
        model = DDPM(unet=unet, cfg=CFG, device=device).to(device)

    return model

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
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'eval'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model-type', type=str, default='VAE', choices=['VAE', 'DDPM'], help='What type of model to use (default: %(default)s)')
    args = parser.parse_args()
    CFG = load_config('configs/config.yaml')
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    
    root_dir = 'data/processed'  
    labels = ['front', 'back', 'shiny'] 
    games = ["red-blue", "gold", "emerald", "firered-leafgreen", "diamond-pearl", "heartgold-soulsilver", "black-white"]
    pokemon_dataset = PokemonDataset(root_dir, labels, games=games)
    train_loader = DataLoader(pokemon_dataset, batch_size=32, shuffle=True)
    _x, _ = next(iter(train_loader))
    save_image(_x, 'input_samples.png')

    if args.mode == 'train':
        print("Training...")
        ### initialize model
        model = build_model(args.model_type, CFG, device)
        ### initialize dataloader
        root_dir = 'data/processed'
        labels = ['front']#, 'back', 'shiny'] 
        games = ["red-blue", "gold", "emerald", "firered-leafgreen"]# "diamond-pearl", "heartgold-soulsilver", "black-white"]
        games = ["emerald", "firered-leafgreen", "diamond-pearl", "heartgold-soulsilver"]
        pokemon_dataset = PokemonDataset(root_dir, labels, games=games)
        # Create a DataLoader
        if args.model_type == 'DDPM':
            pokemon_dataset = PokemonDataset(root_dir, labels, games=games, 
                                             transform=transforms.Compose([
                                             transforms.ToTensor(),       # from [0,255] to range [0.0,1.0]
                                             transforms.Normalize((0.5,), (0.5,))  # range [-1,1]
                                             ]))
            print(f'Normalized train samples!')
            train_loader = DataLoader(pokemon_dataset, batch_size=CFG['training']['batch_size'],
                                       shuffle=True)
        
        else:
            train_loader = DataLoader(pokemon_dataset, batch_size=CFG['training']['batch_size'], shuffle=True)
        
        ### train model
        print("Starting training!")
        trainer = Trainer(model, train_loader, config_path='configs/config.yaml', device=device)
        trainer.train()
        torch.save(model.state_dict(), f=f'{args.model_type}_weights.pt')
        
    if args.mode == 'eval':
        print(f'Evaluating...')
        raise NotImplementedError()

    if args.mode == 'sample':
        import pdb
        model = build_model(args.model_type, CFG, device)
        if os.path.exists(f'{args.model_type}_weights.pt'):
            sd = torch.load(f'{args.model_type}_weights.pt', map_location=device)
            model.load_state_dict(sd)
            print(f'Sampling using weights: {args.model_type}_weights.pt')

        else:
            print(f'Warning! No state dict is loaded for model {args.model_type} when sampling.\nProcedding without loading pretrained weights...')
        
        model.eval()
        with torch.no_grad():
            samples = model.sample(n_samples=32)
            
            # if args.model_type == 'DDPM':
            #     samples = (samples.clamp(-1, 1) + 1) / 2
            #     samples *= 255
        
        if args.model_type == 'VAE':        
            save_image(samples, fp=f'{args.model_type}_samples.png')
        else: ### DDPM
            save_images(denormalize(samples), path='DDPM_samples.png', title='DDPM samples')



        

            
        
    