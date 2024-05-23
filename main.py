import argparse
import os
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from torchvision.utils import save_image
import pdb

from src.data_utils.dataloader import PokemonDataset, PokemonFusionDataset, ResizeSprite
from src.training.trainer import Trainer
from src.models.DDPM import DDPM
from src.models.unet import UNet
from src.models.VAE import VAE, GaussianEncoder, MultivariateGaussianDecoder
from src.models.vae_priors import StdGaussianPrior
from src.utils.misc import load_config
from src.visualizations.functions import denormalize, save_images


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

def build_dataset(dataset_type: str, model_type: str = 'VAE'):
    
    ## define data augmentations
    transform = [transforms.ToTensor()]
    if model_type == 'DDPM':
        transform = [transforms.ToTensor(), # from [0,255] to range [0.0,1.0]
                    transforms.Normalize((0.5,), (0.5,))  # range [-1,1]
                    ]
    
    if dataset_type == 'original':
        org_dir = 'data/processed'  
        labels = ['front', 'shiny'] 
        games = ["emerald", "firered-leafgreen", "diamond-pearl", "heartgold-soulsilver", "black-white"]
        dataset = PokemonDataset(org_dir, labels, games=games, transform=transforms.Compose(transform))
        
    
    elif dataset_type == 'fusion':
        fusion_dir = 'data/fusion'
        fusion_transforms = [transforms.CenterCrop(220), ResizeSprite((64, 64))]
        fusion_transforms.extend(transform)
        dataset = PokemonFusionDataset(fusion_dir, transform=transforms.Compose(fusion_transforms))
        
    
    elif dataset_type == 'all':
        org_dataset = build_dataset('original')
        fusion_dataset = build_dataset('fusion')
        dataset = ConcatDataset([org_dataset, fusion_dataset])

    return dataset

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'eval'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model-type', type=str, default='VAE', choices=['VAE', 'DDPM'], help='What type of model to use (default: %(default)s)')
    parser.add_argument('--data-type', type=str, default='original', choices=['original', 'fusion', 'all'], help='What type of data to use (default: %(default)s)')
    args = parser.parse_args()
    
    CFG = load_config('configs/config.yaml')
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.mode == 'train':
        print("Training...")
        ### initialize model
        model = build_model(args.model_type, CFG, device)
       
        ### initialize dataloader
        dataset = build_dataset(args.data_type, args.model_type)
        train_loader = DataLoader(dataset, batch_size=CFG['training']['batch_size'], shuffle=True)
        
        ### train model
        print("Starting training!")
        trainer = Trainer(model, train_loader, config_path='configs/config.yaml', device=device)
        trainer.train()
        torch.save(model.state_dict(), f='weights/{args.model_type}_weights.pt')
        
    if args.mode == 'eval':
        print(f'Evaluating...')
        raise NotImplementedError()

    if args.mode == 'sample':
        import pdb
        model = build_model(args.model_type, CFG, device)
        if os.path.exists(f'weights/{args.model_type}_weights.pt'):
            sd = torch.load(f'weights/{args.model_type}_weights.pt', map_location=device)
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
            save_image(samples, fp=f'figures/{args.model_type}_samples.png')
        else: ### DDPM
            save_images(denormalize(samples), path=f'figures/{args.model_type}_samples.png', title='DDPM samples')




        

            
        
    