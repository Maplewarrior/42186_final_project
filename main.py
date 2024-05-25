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
from src.models.vae_priors import StdGaussianPrior, MixtureOfGaussiansPrior, VampPrior
from src.utils.misc import load_config
from src.visualizations.functions import denormalize, save_images
from src.visualizations.plot_loss import plot_loss
import uuid

import wandb

def build_model(model_type: str, CFG: dict, device: str, vae_prior_type: str = 'std_gauss'):
    H = CFG['data']['H']
    W = CFG['data']['W']
    C = CFG['data']['channels']
    D = CFG['VAE']['D']
    num_components = CFG['VAE']['num_components']

    if model_type == 'VAE':       
        encoder = GaussianEncoder(H, W, D)
        decoder = MultivariateGaussianDecoder(H, W, D)

        # Make prior depending on the type chosen
        if vae_prior_type == 'std_gauss': 
            prior = StdGaussianPrior(D)
        elif vae_prior_type == 'mog':
            prior = MixtureOfGaussiansPrior(num_components=CFG['VAE']['num_components'], latent_dim=D)
        elif vae_prior_type == 'vamp':
            prior = VampPrior(num_components=num_components, H=H, W=W, C=C, encoder=encoder)

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
    parser.add_argument('--p-uncond', type=float, default=None, help='probability of doing unconditional sampling when training DDPM model. If None the value in config is kept.')
    parser.add_argument('--vae-prior', type=str, default='std_gauss', choices=['std_gauss', 'mog', 'vamp'], help='What type of prior to use for VAE (default: %(default)s)')
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()

    CFG = load_config('configs/config.yaml')
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.p_uncond != None:
        CFG['DDPM']['p_uncond'] = args.p_uncond
    
    
    if args.mode == 'train':
        print("Training...")
        ### initialize model
        model = build_model(args.model_type, CFG, device, vae_prior_type=args.vae_prior)
       
        ### initialize dataloader
        dataset = build_dataset(args.data_type, args.model_type)
        # define dataloader --> NOTE: drop_last=True because CFG implementation expects a fixed batch size
        train_loader = DataLoader(dataset, batch_size=CFG['training']['batch_size'], shuffle=True, drop_last=True)


        uid = uuid.uuid4()
        weight_filename = f'weights/{args.model_type}_weights_{uid}.pt'

        ### train model
        print("Starting training!")
        if args.wandb:
            wand_cfg = {
                "data-type": args.data_type,
                "model-type": args.model_type,
                "batch-size": CFG['training']['batch_size'],
                "weight-file": weight_filename,
                "uid": uid,
                "optimizer": CFG["training"]["optimization"]["optimizer_name"],
                "learning-rate": CFG["training"]["optimization"][CFG["training"]["optimization"]["optimizer_name"]]["lr"],
            }
            if args.model_type == 'VAE':
                wand_cfg["vae-prior"] = args.vae_prior
                wand_cfg["num-components"] = CFG['VAE']['num_components']
                wand_cfg["latent-dim"] = CFG['VAE']['D']
            if args.model_type == 'DDPM':
                wand_cfg["p-uncond"] = CFG['DDPM']['p_uncond']
                wand_cfg["T"] = CFG['DDPM']['T']
            # start a new wandb run to track this script
            wandb.init(
                # set the wandb project where this run will be logged
                project="MBMLexam",
                # track hyperparameters and run metadata
                config=wand_cfg
            )

        if args.wandb:
            trainer = Trainer(model, train_loader, config_path='configs/config.yaml', device=device, uuid=uid, wandb=wandb)
        else:
            trainer = Trainer(model, train_loader, config_path='configs/config.yaml', device=device, uuid=uid)
        losses = trainer.train()
        plot_loss(losses, args.model_type) # save plot of losses
        torch.save(model.state_dict(), weight_filename)
        wandb.finish()

    if args.mode == 'eval':
        print(f'Evaluating...')
        raise NotImplementedError()

    if args.mode == 'sample':
        import pdb
        model = build_model(args.model_type, CFG, device, vae_prior_type=args.vae_prior)
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




        

            
        
    