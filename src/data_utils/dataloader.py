import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from src.data_utils.metadata import PokemonMetaData
import torch
import pdb

def divide_by_255(x):
    return x / 255.0

# Function to extract the sorting key
def extract_key(filename):
    # Remove the file extension
    filename = os.path.splitext(filename)[0]
    # Check if the filename is an integer
    if filename.split('-')[0].isdigit():
        # Return a tuple with 0 to ensure integers come first and the integer value
        return (0, int(filename.split('-')[0]))
    else:
        # Return a tuple with 1 to ensure strings come after integers and the filename
        return (1, filename)

class PokemonDataset(Dataset):
    def __init__(self, root_dir, labels=None, games=None, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            labels (list): List of subfolder names to be used as labels. E.g., ['front', 'back', 'shiny']
            games (list): List of game names to include. E.g., ['red-blue', 'gold', 'emerald']
        """
        self.metadata = PokemonMetaData()
        self.root_dir = root_dir
        self.labels = labels if labels is not None else ['front', 'back', 'shiny']
        self.games = games  # None means include all games
        self.transform = transform

        if transform is None:
            self.transform = transforms.Compose([
                # Add more transforms here as needed
                transforms.ToTensor()
            ])
        self.image_paths = []
        self.image_labels = []

        # Traverse the directory structure
        for gen_folder in os.listdir(root_dir):
            gen_path = os.path.join(root_dir, gen_folder)
            if os.path.isdir(gen_path):
                if "main-sprites" in os.listdir(gen_path):
                    old_gen_path = gen_path
                    gen_path = os.path.join(gen_path, 'main-sprites')
                for version_folder in os.listdir(gen_path):
                    # If games filter is set, skip versions not in the list
                    if self.games is not None and version_folder not in self.games:
                        continue
                    
                    version_path = os.path.join(gen_path, version_folder)
                    for label in self.labels:
                        if label == "front" and "main-sprites" in os.listdir(old_gen_path):
                            label_path = version_path
                        else:
                            label_path = os.path.join(version_path, label)
                        if os.path.isdir(label_path):
                            # sort label_path by id
                            for img_file in sorted(os.listdir(label_path), key=extract_key):
                                if img_file.endswith('.png'):
                                    # pdb.set_trace()
                                    image_name = img_file.split('.')[0]
                                    # imagename is digit?
                                    if not image_name.isdigit() or int(image_name) == 0:
                                        continue
                                    
                                    type_name = self.metadata.get_type_by_id(int(image_name), numeric=True)
                                        
                                    self.image_labels.append(type_name)
                                    self.image_paths.append(os.path.join(label_path, img_file))
                                    # self.image_labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.image_labels[idx]
        image = Image.open(img_path).convert('RGBA')
        # Check if image has transparency
        if image.mode == 'RGBA':
            # Create a white background image
            white_background = Image.new('RGBA', image.size, (255, 255, 255, 255))
            # Paste the image on top of the white background
            image = Image.alpha_composite(white_background, image)
            # Convert back to RGB
            image = image.convert('RGB')

        image = self.transform(image)
        return image, label

class PokemonFusionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images organized in subfolders.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.metadata = PokemonMetaData()
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_to_idx = self.metadata.types_dict
        
        self._prepare_dataset()

    def _prepare_dataset(self):
        # Traverse the directory structure
        for label in os.listdir(self.root_dir):
            label_dir = os.path.join(self.root_dir, label)
            if os.path.isdir(label_dir):
                for img_name in os.listdir(label_dir):
                    img_path = os.path.join(label_dir, img_name)
                    if os.path.isfile(img_path) and img_path.endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(img_path)
                        self.labels.append(self.label_to_idx[label])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):        
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGBA')  # Open image in RGBA mode to handle transparency
        label = self.labels[idx]

        # Check if image has transparency
        if image.mode == 'RGBA':
            # Create a white background image
            white_background = Image.new('RGBA', image.size, (255, 255, 255, 255))
            # Paste the image on top of the white background
            image = Image.alpha_composite(white_background, image)
            # Convert back to RGB
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label

# Custom transformation to resize sprites while preserving clarity
class ResizeSprite:
    def __init__(self, size):
        self.size = size

    def __call__(self, image):
        return image.resize(self.size, Image.NEAREST)


class CenterElementCrop:
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, img):
        width, height = img.size
        crop_width, crop_height = self.crop_size

        center_x, center_y = width // 2, height // 2
        left = max(center_x - crop_width // 2, 0)
        top = max(center_y - crop_height // 2, 0)
        right = min(center_x + crop_width // 2, width)
        bottom = min(center_y + crop_height // 2, height)

        return img.crop((left, top, right, bottom))

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    # ======================== Pokemon dataset test ========================

    # Usage
    root_dir = 'data/raw'  
    # labels = ['front', 'back', 'shiny'] 
    labels = ['front', 'shiny']
    games1 = [ "emerald", "firered-leafgreen"]
    games2 = ["diamond-pearl", "heartgold-soulsilver"]
    
    trans1 = transforms.Compose([
        transforms.Pad(4, fill=255, padding_mode='constant'), # Gen emerald, firered-leafgreen
        ResizeSprite((64, 64)),
        transforms.ToTensor()
    ])

    padded_dataset = PokemonDataset(root_dir, labels, games=games1, transform=trans1)

    trans2 = transforms.Compose([
        ResizeSprite((64, 64)),
        transforms.ToTensor()
    ])

    unpadded_dataset = PokemonDataset(root_dir, labels, games2, transform=trans2)

    # merge into one dataset
    pokemon_dataset = torch.utils.data.ConcatDataset([padded_dataset, unpadded_dataset])


    # pokemon_dataset = PokemonDataset(root_dir, labels, games=games, transform=trans)

    # Create a DataLoader
    dataloader = DataLoader(pokemon_dataset, batch_size=16, shuffle=True)

    

    # Example of iterating over DataLoader
    for images_orig, labels in dataloader:
        # Process images and labels here
        print(images_orig.shape, labels)
        break

    # ======================== Pokemon fusion dataset test ========================


    # Sample usage
    root_dir = 'data/fusion'
    trans = transforms.Compose([
        # transforms.Resize((64, 64)),
        # crop the image to 164x164
        transforms.CenterCrop(220),
        # CenterElementCrop((220, 220)),
        ResizeSprite((64, 64)),
        # transforms.RandomHorizontalFlip(p=0.0),  # This line is optional
        transforms.ToTensor()
    ])

    fusion_dataset = PokemonFusionDataset(root_dir, transform=trans)

    # Create a DataLoader
    dataloader = DataLoader(fusion_dataset, batch_size=16, shuffle=False)

    # Example of iterating over DataLoader
    for images, labels in dataloader:
        # Process images and labels here
        print(images.shape, labels)
        break

    print("Fusion dataset", len(fusion_dataset))
    print("Pokemon dataset", len(pokemon_dataset))
    print("Total pokemons", len(fusion_dataset)+len(pokemon_dataset))

    # merge images_orig and images
    images = torch.cat((images_orig, images), 0)

    fig, axs = plt.subplots(4, 8, figsize=(10, 10))
    axs = axs.flatten()
    for i in range(16*2):
        ax = axs[i]
        ax.imshow(images[i].permute(1, 2, 0))
        ax.axis('off')

    # plt.title('Pokemon Fusion Dataset')

    plt.tight_layout()
    plt.show()
