import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

def divide_by_255(x):
    return x / 255.0

class PokemonDataset(Dataset):
    def __init__(self, root_dir, labels=None, games=None, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            labels (list): List of subfolder names to be used as labels. E.g., ['front', 'back', 'shiny']
            games (list): List of game names to include. E.g., ['red-blue', 'gold', 'emerald']
        """
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
                for version_folder in os.listdir(gen_path):
                    # If games filter is set, skip versions not in the list
                    if self.games is not None and version_folder not in self.games:
                        continue
                    
                    version_path = os.path.join(gen_path, version_folder)
                    for label in self.labels:
                        label_path = os.path.join(version_path, label)
                        if os.path.isdir(label_path):
                            for img_file in os.listdir(label_path):
                                if img_file.endswith('.png'):
                                    self.image_paths.append(os.path.join(label_path, img_file))
                                    self.image_labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.image_labels[idx]
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image, label

class PokemonFusionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images organized in subfolders.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.label_to_idx = {}
        
        self._prepare_dataset()

    def _prepare_dataset(self):
        # Traverse the directory structure
        for label_idx, label in enumerate(os.listdir(self.root_dir)):
            label_dir = os.path.join(self.root_dir, label)
            if os.path.isdir(label_dir):
                self.label_to_idx[label] = label_idx
                for img_name in os.listdir(label_dir):
                    img_path = os.path.join(label_dir, img_name)
                    if os.path.isfile(img_path) and img_path.endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append(img_path)
                        self.labels.append(label_idx)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

if __name__ == '__main__':

    import matplotlib.pyplot as plt

    # Usage
    root_dir = 'data/processed'  
    labels = ['front', 'back', 'shiny'] 

    games = ["red-blue", "gold", "emerald", "firered-leafgreen", "diamond-pearl", "heartgold-soulsilver", 
             "black-white"]

    transform = transforms.Compose([
                # Add more transforms here as needed
                transforms.ToTensor(),
                transforms.Lambda(divide_by_255)
            ])

    pokemon_dataset = PokemonDataset(root_dir, labels, games=games, transform=transform)

    # Create a DataLoader
    dataloader = DataLoader(pokemon_dataset, batch_size=32, shuffle=True)

    # Example of iterating over DataLoader
    for images, labels in dataloader:
        # Process images and labels here
        print(images.shape, labels)
        break


    trans = transforms.Compose([
        # resize to 64x64
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    fusion_dataset = PokemonFusionDataset('data/fusion', transform=trans)
    dataloader = DataLoader(fusion_dataset, batch_size=4, shuffle=True)

    for images, labels in dataloader:
        print(images.shape, labels)
        break
