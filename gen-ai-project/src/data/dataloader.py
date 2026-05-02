import sys
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T 

def get_transforms_vae(image_size):
    """Retorna as transformações padrão para as imagens."""
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(image_size),
        T.ToTensor(),  # converte para [0,1]
    ])

def get_transforms_gan(image_size):
    """Transformações para a GAN: coloca os píxeis no intervalo [-1, 1]"""
    return T.Compose([
        T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(image_size),
        T.ToTensor(),  
        T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

class HFDatasetTorch(Dataset):
    def __init__(self, hf_split, transform=None, indices=None):
        self.ds = hf_split
        self.transform = transform
        self.indices = list(range(len(hf_split))) if indices is None else list(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        ex = self.ds[real_idx]
        img = ex["image"]
        y = int(ex["label"])
        x = self.transform(img) if self.transform else img
        return x, y, real_idx