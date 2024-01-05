import os
import glob
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import cv2

import sys
sys.path.append(".")
from datasets.transforms import pad
from pathlib import Path


class ImageDataset(Dataset): 
    def __init__(self, image_paths, transform=None): 
        self.image_paths = image_paths
        self.transform = transform
  
    def __len__(self): 
        return len(self.image_paths)
    
    def __getitem__(self, idx): 
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (500, 375))
  
        if self.transform:
            image = self.transform(image)
        return image

if __name__ == '__main__':  
    image_folder = './transformed'
    extension='jpg'

    pattern = os.path.join(image_folder, '**', f'*.{extension}')
    image_paths = glob.glob(pattern, recursive=True)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = ImageDataset(image_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=len(image_paths), shuffle=False)
    images = next(iter(dataloader))

    mean = images.mean(dim=(0, 2, 3))
    std = images.std(dim=(0, 2, 3))
    print("norm_mean:", mean.numpy())
    print("norm_std:", std.numpy())


    """
    ## Another way to do this

    dataset = ImageDataset(image_paths, transform=transform)
    dataloader = DataLoader(dataset, batch_size=len(image_paths), shuffle=False)

    psum = torch.zeros(3)
    psum_sq = torch.zeros(3)

    for image in tqdm(dataloader):
        psum += image.sum(axis = [0, 2, 3])
        psum_sq += (image ** 2).sum(axis = [0, 2, 3])

    count = len(image_paths) * 500 * 375

    mean = psum / count
    var = (psum_sq / count) - (mean ** 2)
    std = torch.sqrt(var)

    print("norm_mean:", mean.numpy())
    print("norm_std:", std.numpy())
    """