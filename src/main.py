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
from src.data_utils.metadata import PokemonMetaData
import uuid
from tqdm import tqdm

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
        unet = UNet(img_size=H, c_in=C, c_out=C, device=device, n_classes=18).to(device)
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
        org_dir = 'data/raw'  
        labels = ['front', 'shiny'] 

        # Split into two datasets
        games1 = [ "emerald", "firered-leafgreen"]
        games2 = ["diamond-pearl", "heartgold-soulsilver"]
    
        # One dataset with padding
        t1 = [transforms.Pad(4, fill=255, padding_mode='constant'), ResizeSprite((64, 64))]
        t1.extend(transform)
        dataset1 = PokemonDataset(org_dir, labels, games=games1, transform=transforms.Compose(t1))
        # One dataset without padding
        t2 = [ResizeSprite((64, 64))]
        t2.extend(transform)
        dataset2 = PokemonDataset(org_dir, labels, games2, transform=transforms.Compose(t2))

        # Merge into one dataset
        dataset = torch.utils.data.ConcatDataset([dataset1, dataset2])

    
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
    parser.add_argument('mode', type=str, default='train', choices=['train', 'sample', 'eval', 'sample-cond'], help='what to do when running the script (default: %(default)s)')
    parser.add_argument('--model-type', type=str, default='VAE', choices=['VAE', 'DDPM'], help='What type of model to use (default: %(default)s)')
    parser.add_argument('--data-type', type=str, default='original', choices=['original', 'fusion', 'all'], help='What type of data to use (default: %(default)s)')
    parser.add_argument('--p-uncond', type=float, default=None, help='probability of doing unconditional sampling when training DDPM model. If None the value in config is kept.')
    parser.add_argument('--vae-prior', type=str, default='std_gauss', choices=['std_gauss', 'mog', 'vamp'], help='What type of prior to use for VAE (default: %(default)s)')
    parser.add_argument('--load-weights', type=str, default=None, help='Path to weights to load for model')
    parser.add_argument('--num-samples', type=int, default=32, help='Number of samples to generate')
    parser.add_argument('--sample-batch-size', default=32, type=int, help='Batch size for sampling')
    parser.add_argument('--class-cond', type=str, default=None, help='Which class to use for conditional sampling for the DDPM model. If set to None, unconditional sampling is performed (default: %(default)s)')
    parser.add_argument('--wandb', action='store_true')
    args = parser.parse_args()

    CFG = load_config('configs/config.yaml')
    device = ('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.p_uncond != None:
        CFG['DDPM']['p_uncond'] = args.p_uncond

    # Fix to overwrite batch size and n_epochs for VAE and DDPM
    if args.model_type == 'VAE':
        CFG['training']['batch_size'] = CFG['VAE']['batch_size']
        CFG['training']['n_epochs'] = CFG['VAE']['n_epochs']
    elif args.model_type == 'DDPM':
        CFG['training']['batch_size'] = CFG['DDPM']['batch_size']
        CFG['training']['n_epochs'] = CFG['DDPM']['n_epochs']
    
    
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
            trainer = Trainer(model, train_loader, config=CFG, device=device, uuid=uid, wandb=wandb)
        else:
            trainer = Trainer(model, train_loader, config=CFG, device=device, uuid=uid)
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

        model_path = args.load_weights
        if os.path.exists(model_path):
            sd = torch.load(model_path, map_location=device)
            # If the state dict is nested
            if type(sd) == dict:
                uid = model_path.split("/")[-2]
                sd = sd['model']
            else:
                uid = model_path.split('_')[-1].split('.')[0]

            loaded = model.load_state_dict(sd)
            print(f'Sampling using weights: {model_path}')
        else:
            print(f'Warning! No state dict is loaded for model {args.model_type} when sampling.\nProcedding without loading pretrained weights...')
        
        metadata = PokemonMetaData(types_path='data/types.csv')

        num_samples = args.num_samples
        batch_size = args.sample_batch_size
        # Generate a total of num_samples samples
        n_batches = num_samples // batch_size
        model.eval()
        for i in range(n_batches):
            with torch.no_grad():
                y = None
                if args.model_type == 'DDPM':
                    class2idx = metadata.types_dict
                    if (args.class_cond != None) and (args.class_cond in class2idx.keys()):
                        y = torch.zeros((batch_size, CFG['data']['n_classes']), device=model.device)
                        y[:, class2idx[args.class_cond]] = 1
                    elif i == 0:
                        print("Sampling DDPM unconditionally!")
                        

                samples = model.sample(n_samples=batch_size, y=y)

            if args.model_type == 'DDPM':
                samples = denormalize(samples)

            # Now save the samples
            os.makedirs(f'samples/{args.model_type}/{uid}/', exist_ok=True)
            torch.save(samples, f'samples/{args.model_type}/{uid}/{batch_size}_samples_{i}.pt')

    if args.mode == "sample-cond": # samples DDPM conditionally using empirical distribution of classes
        metadata = PokemonMetaData(types_path='data/types.csv')
        model = build_model(args.model_type, CFG, device, vae_prior_type=args.vae_prior)

        model_path = args.load_weights
        if os.path.exists(model_path):
            sd = torch.load(model_path, map_location=device)
            # If the state dict is nested
            if type(sd) == dict:
                uid = model_path.split("/")[-2]
                sd = sd['model']
            else:
                uid = model_path.split('_')[-1].split('.')[0]

            loaded = model.load_state_dict(sd)
            print(f'Sampling using weights: {model_path}')
        else:
            print(f'Warning! No state dict is loaded for model {args.model_type} when sampling.\nProcedding without loading pretrained weights...')
        # get class distribution

        type_counts = {k: 0 for k in metadata.types_dict.keys()}
        idx_to_label = {v: k for k, v in metadata.types_dict.items()}
        dataset = build_dataset(args.data_type, args.model_type)
        for _, lab in dataset:
            type_counts[idx_to_label[lab]] += 1
        
        num_samples = args.num_samples

        # calculate how many samples to generate for each class for a total of num_samples
        samples_per_class = {k: int(num_samples * v / len(dataset)) for k, v in type_counts.items()}

        batch_size = args.sample_batch_size

        # how many batches to generate per class
        batches_per_class = {k: v // batch_size for k, v in samples_per_class.items()}

        for type, n_batches in batches_per_class.items():
                if type in ["Bug", "Dark", "Dragon", "Electric", "Fairy", "Fighting", "Fire", "Flying", "Ghost", "Grass", "Ground", "Ice", "Normal"]:
                    continue
                model.eval()
                for i in tqdm(range(n_batches), desc=f"Processing {type} batches"):
                    with torch.no_grad():
                        if args.model_type == 'DDPM':
                            class2idx = metadata.types_dict
                            if (type != None) and (type in class2idx.keys()):
                                y = torch.zeros((batch_size, CFG['data']['n_classes']), device=model.device)
                                y[:, class2idx[type]] = 1
                            else:
                                print("Sampling DDPM unconditionally!")
                                y = None
                            samples = model.sample(n_samples=batch_size, y=y)

                    if args.model_type == 'DDPM':
                        samples = denormalize(samples)

                        # TODO If DDPM conditional, we need to add the condition to the samples

                    # Now save the samples
                    os.makedirs(f'samples_cond/{args.model_type}/{uid}/{type}/', exist_ok=True)
                    torch.save(samples, f'samples_cond/{args.model_type}/{uid}/{type}/{batch_size}_samples_{i}.pt')