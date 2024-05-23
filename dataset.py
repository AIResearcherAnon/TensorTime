import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch

class ImageNet32Dataset(Dataset):
    def __init__(self, directory_path, transform=None):
        """
        Args:
            directory_path (string): Directory path containing the .npz files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.directory_path = directory_path
        self.transform = transform
        self.images = []
        self.labels = []

        # Load data from all .npz files in the directory
        for file_name in sorted(os.listdir(directory_path)):
            if file_name.endswith('.npz'):
                data_path = os.path.join(directory_path, file_name)
                data = np.load(data_path)
                # Reshape the images to the format (num_samples, 3, 32, 32) and append
                self.images.append(data['data'].reshape((-1, 3, 32, 32)).astype(np.uint8))
                self.labels.append(data['labels'])
        
        # Concatenate all loaded datasets
        self.images = np.concatenate(self.images, axis=0)
        self.labels = np.concatenate(self.labels, axis=0)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx] - 1  # Adjust labels to start from 0
        image = image.transpose((1, 2, 0))  # Convert to HWC format for transforms
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
