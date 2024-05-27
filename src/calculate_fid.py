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
    # import argparse
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--sample-folder', type=str, help='path to the folder containing the sampled images')
    # parser.add_argument('--device', type=str, default="cpu", help='device to use (default: %(default)s)')

    # args = parser.parse_args()
    # device = torch.device(args.device)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Define the transformation pipeline
    trans = transforms.Compose([
        transforms.Resize(299),
        transforms.CenterCrop(299),
        # transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.ToTensor(),  # Convert images to tensors in the range [0, 1]
    ])


    root_folder = "./data/fusion"
    real_dataset = PokemonFusionDataset(root_folder, transform=trans)
    real_dataset = build_dataset('all')


    trans = transforms.Compose([
        # transform to PIL
        transforms.ToPILImage(),
        transforms.Resize(299),
        transforms.CenterCrop(299),
        # make it between 0 and 1
        transforms.ToTensor()
    ])

    generated_dataset = SamplesDataset("samples/VAE/f05c86aa-7700-4020-bd6d-51f0a99dc598", transform=trans)
    
    # # Create indices for each subset
    # indices_label_0 = [i for i, (_, label) in enumerate(real_dataset) if label == 0]
    # indices_label_1 = [i for i, (_, label) in enumerate(real_dataset) if label == 6]

    # # Create subsets using the indices
    # subset_label_0 = Subset(real_dataset, indices_label_0)
    # subset_label_1 = Subset(real_dataset, indices_label_1)

    # Create DataLoaders for the generated and real images
    generated_images_dataloader = DataLoader(real_dataset, batch_size=32, shuffle=False)
    real_images_dataloader = DataLoader(generated_dataset, batch_size=32, shuffle=False)

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
