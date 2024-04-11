import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

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
        model = VAE(encoder, decoder, prior)
    
    elif model_type == 'DDPM':
        unet = UNet(img_size=H, c_in=C, c_out=C, device=device)
        model = DDPM(unet=unet, cfg=CFG)

    return model


    
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'eval'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model-type', type=str, default='VAE', choices=['VAE', 'DDPM'], help='What type of model to use (default: %(default)s)')
    args = parser.parse_args()
    CFG = load_config('configs/config.yaml')
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    
    # root_dir = 'data/processed'  
    # labels = ['front', 'back', 'shiny'] 
    # games = ["red-blue", "gold", "emerald", "firered-leafgreen", "diamond-pearl", "heartgold-soulsilver", "black-white"]
    # pokemon_dataset = PokemonDataset(root_dir, labels, games=games)
    # train_loader = DataLoader(pokemon_dataset, batch_size=32, shuffle=True)
    # _x, _ = next(iter(train_loader))
    # save_image(_x, 'input_samples.png')

    if args.mode == 'train':
        ### initialize model
        model = build_model(args.model_type, CFG, device)
        ### initialize dataloader
        root_dir = 'data/processed'
        labels = ['front']#, 'back', 'shiny'] 
        games = ["red-blue", "gold", "emerald", "firered-leafgreen"]# "diamond-pearl", "heartgold-soulsilver", "black-white"]
        pokemon_dataset = PokemonDataset(root_dir, labels, games=games)
        # Create a DataLoader
        train_loader = DataLoader(pokemon_dataset, batch_size=CFG['training']['batch_size'], shuffle=True)
        
        ### train model
        trainer = Trainer(model, train_loader, config_path='configs/config.yaml', device=device)
        trainer.train()
        torch.save(model.state_dict(), f=f'{args.model_type}_weights.pt')
        
    if args.mode == 'eval':
        print(f'Evaluating...')


    if args.mode == 'sample':
        model = build_model(args.model_type, CFG, device)
        if os.path.exists('{model_type}_weights.pt'.format(model_type=args.model_type)):
            sd = torch.load(f'{args.model_type}_weights.pt', map_location='cpu')
            model.load_state_dict(sd)
        else:
            print(f'Warning! No state dict is loaded for model {args.model_type} when sampling.\nProcedding without loading pretrained weights...')
        model.eval()
        with torch.no_grad():
            samples = model.sample(n_samples=32)
            print(samples.max())
        save_image(samples, fp=f'{args.model_type}_samples.png')
        
    