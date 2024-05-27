# %%
import torch
from torchvision.transforms import functional as TF
from torcheval.metrics import FrechetInceptionDistance
from torchvision.models import inception_v3, Inception_V3_Weights
import os
from tqdm import tqdm
from src.data_utils.dataloader import PokemonFusionDataset, PokemonDataset
from src.data_utils.samples_dataloader import SamplesDataset
from src.main import build_dataset
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
# %%
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--samples-folder', type=str, help='Path to the folder containing the samples')

    args = parser.parse_args()

    # device = torch.device(args.device)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Get thte real dataset
    real_dataset = build_dataset('all')


    trans = transforms.Compose([
        # transform to PIL
        transforms.ToPILImage(),
        transforms.Resize(299),
        transforms.CenterCrop(299),
        # make it between 0 and 1
        transforms.ToTensor()
    ])

    samples_folder = args.samples_folder
    generated_dataset = SamplesDataset(samples_folder, transform=trans, device=device)
    
    # Create DataLoaders for the generated and real images
    generated_images_dataloader = DataLoader(generated_dataset, batch_size=32, shuffle=False)
    real_images_dataloader = DataLoader(real_dataset, batch_size=32, shuffle=False)

    # Load the pretrained Inception v3 model
    inception_model = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1).to(device)
    inception_model.fc = torch.nn.Identity()
    inception_model.eval()

    # Initialize the FID metric with the Inception V3 model as the feature extractor
    fid_metric = FrechetInceptionDistance(device=device)

    with torch.no_grad():
        # Process generated images with tqdm progress bar
        for generated_images in tqdm(generated_images_dataloader, desc="Processing Generated Images"):
            if isinstance(generated_images, (list, tuple)):
                generated_images = generated_images[0]
            generated_images = generated_images.to(device)
            fid_metric.update(generated_images, is_real=False)

        # Process real images with tqdm progress bar
        for real_images in tqdm(real_images_dataloader, desc="Processing Real Images"):
            if isinstance(real_images, (list, tuple)):
                real_images = real_images[0]
            real_images = real_images.to(device)
            fid_metric.update(real_images, is_real=True)

    # Compute and print the FID score

    # move to device cpu if mps
    # NOTE: This is a workaround as the FID metric does not support MPS devices
    if device == "mps":
        fid_metric.to("cpu")
    fid_score = fid_metric.compute()
    print("FID Score:", fid_score)
# %%
