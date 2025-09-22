import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from bsccm import BSCCM
from typing import List, Tuple
import json

class BSCCMDataset(Dataset):
    """
    Dataset for training a model to predict QPI from only two DPC images.
    """

    def __init__(self, 
                 dataset_path: str = "/share/monakhova/Cassandra_data/UQNet_proj/bsccm", 
                 input_channels: List[str] = ['DPC_Left', 'DPC_Right'],
                 transform=None):
        self.dataset = BSCCM(dataset_path, cache_index=True)
        self.indices = self.dataset.get_indices()
        self.input_channels = input_channels
        self.transform = transform
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        index = self.indices[idx]

        input_images = []
        for channel in self.input_channels:
            image = self.dataset.read_image(index, channel=channel)
            input_images.append(image)
        input_image_f32 = [arr.astype(np.float32) for arr in input_images]
        input_tensor = torch.tensor(np.stack(input_image_f32, axis=0), dtype=torch.float32)

        qpi = self.dataset.read_image(index, channel='dpc')
        qpi_tensor = torch.tensor(qpi, dtype=torch.float32).unsqueeze(0)

        if self.transform:
            input_tensor = self.transform(input_tensor)

        return input_tensor, qpi_tensor
    
    def get_indices(self):
        return self.indices.copy()
        
def get_bsccm_dataloaders(
        dataset_path: str = "/share/monakhova/Cassandra_data/UQNet_proj/bsccm",
        batch_size: int = 16,
        input_channels: List[str] = ['DPC_Left', 'DPC_Right'],
        validation_split: float = 0.1,
        calib_split: float = 0.2,
        test_split: float = 0.1,
        transform=None,
        num_workers: int = 4,
        pin_memory: bool = True,):
    
    dataset = BSCCMDataset(dataset_path, input_channels, transform)
    indices = dataset.get_indices()
    np.random.shuffle(indices)
    
    dataset_size = len(dataset)
    test_size = int(np.floor(test_split * dataset_size))
    validation_size = int(np.floor(validation_split * dataset_size))
    calib_size = int(np.floor(calib_split * dataset_size))
    train_size = dataset_size - test_size - validation_size - calib_size

    train_indices = indices[:train_size]
    validation_indices = indices[train_size:train_size + validation_size]
    test_indices = indices[train_size + validation_size:train_size + validation_size + test_size]
    calib_indices = indices[train_size + validation_size + test_size:]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    validation_dataset = torch.utils.data.Subset(dataset, validation_indices)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=pin_memory)
    validation_loader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=pin_memory)
    
    return train_loader, validation_loader, calib_indices, test_indices


if __name__ == "__main__":
    dataset = BSCCMDataset()
    train_loader, test_loader, calib_indices, test_indices = get_bsccm_dataloaders()
    print(len(train_loader.dataset), len(test_loader.dataset), len(calib_indices), len(test_indices))