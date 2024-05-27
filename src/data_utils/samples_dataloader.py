import os
import torch
from torch.utils.data import Dataset
from src.data_utils.metadata import PokemonMetaData
import pdb

class SamplesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.data_files = []
        self.labels = []
        self.metadata = PokemonMetaData()
        self.label_to_idx = self.metadata.types_dict
        self.transform = transform
        self._load_files()

    def _load_files(self):
        for root, dirs, files in os.walk(self.root_dir):
            for file in files:
                if file.endswith('.pt'):
                    label = os.path.basename(root)
                    file_path = os.path.join(root, file)
                    batch_size = int(file.split('_')[0])
                    data = torch.load(file_path)
                    if len(data) != batch_size:
                        raise ValueError(f"Batch size mismatch in file {file_path}")
                    for i in range(batch_size):
                        self.data_files.append(data[i])
                        # translate label name to numeric

                        # if label has a uid pattern name
                        if "-" in label:
                            self.labels.append(-1) # NOTE we are not interested in the label if it has a uid pattern
                        elif label.isdigit():
                            self.labels.append(int(label))
                        else:
                            self.labels.append(self.label_to_idx[label])

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data = self.data_files[idx]
        label = self.labels[idx]

        if self.transform:
            data = self.transform(data)
        return data, label
    
if __name__ == "__main__":
    vae_dataset = SamplesDataset('samples/VAE/f05c86aa-7700-4020-bd6d-51f0a99dc598')
    data_loader = torch.utils.data.DataLoader(vae_dataset, batch_size=10, shuffle=True)

    pdb.set_trace()

