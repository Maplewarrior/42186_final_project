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
