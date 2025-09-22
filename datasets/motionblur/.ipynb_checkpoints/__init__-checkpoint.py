from pathlib import Path
from typing import Literal, List

import cv2
import imageio.v3 as iio
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from datasets.motionblur.motionblur import Kernel
from datasets.utils import normalize


class MotionBlurDataset(Dataset):
    def __init__(self, data_root: str = '/share/monakhova/Cassandra_data/UQNet_proj/motionblur',
                 n_blur_samples: int = 5,
                 kernel_min: int = 5,
                 kernel_max: int = 20,
                 intensity: float = 0.1,
                 transform = None, 
                 normalization: Literal[None, 'standard', 'min-max'] = None,
                 kernels: List[int] = None,):
        self.data_root = Path(data_root)
        assert self.data_root.exists(), f"Data root {self.data_root} does not exist."
        self.image_folder = self.data_root / 'mirflickr'
        self.image_paths = list(self.image_folder.glob('*.jpg'))
        if kernels is not None:
            assert isinstance(kernels, list) and all(isinstance(k, int) for k in kernels), \
                "Kernels must be a list of integers."
            assert len(kernels) > 0, "Kernels list must not be empty."
            self.kernels = kernels
            self.n_blur_samples = len(kernels)
        else:
            self.n_blur_samples = n_blur_samples
            self.kernel_min = kernel_min
            self.kernel_max = kernel_max
            self.kernels = None
        self.intensity = intensity
        self.transform = transform
        self.normalization = normalization
        print(f"Found {len(self.image_paths)} images in {self.image_folder}")

    def __len__(self):
        return len(self.image_paths) * self.n_blur_samples
    
    def __getitem__(self, index):
        image_idx = index // self.n_blur_samples
        image_path = self.image_paths[image_idx]

        gt = iio.imread(image_path)
        h, w = gt.shape[:2]
        side = min(h, w) 
        top, left  = (h - side) // 2, (w - side) // 2
        gt = gt[top:top + side, left:left + side]
        gt = cv2.resize(gt, (256, 256), interpolation=cv2.INTER_LINEAR)
        
        if self.kernels is not None:
            kernel_size = self.kernels[index % self.n_blur_samples]
        else:
            assert self.kernel_min <= self.kernel_max, "kernel_min must be less than or equal to kernel_max."
            assert self.kernel_min > 0, "kernel_min must be greater than 0."
            kernel_size = np.random.randint(self.kernel_min, self.kernel_max + 1)
        kernel = Kernel(size=(kernel_size, kernel_size), intensity=self.intensity)
        
        image = np.array(kernel.applyTo(gt.copy(), keep_image_dim=True))
        image, gt = rgb2gray(image), rgb2gray(gt)
        input_tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0)
        gt_tensor = torch.tensor(gt, dtype=torch.float32).unsqueeze(0)
        if self.transform:
            input_tensor = self.transform(input_tensor)
        if self.normalization:
            input_tensor = normalize(input_tensor, type=self.normalization, per_pixel=False)
            gt_tensor = normalize(gt_tensor, type=self.normalization, per_pixel=False)
        
        return input_tensor, gt_tensor

def rgb2gray(rgb: np.ndarray) -> np.ndarray:
    return np.dot(rgb[...,:3], [0.2126, 0.7152, 0.0722])

def get_motionblur_dataloaders(data_root: str = '/share/monakhova/Cassandra_data/UQNet_proj/motionblur',
                                batch_size: int = 16,
                                n_blur_samples: int = 5,
                                kernel_min: int = 5,
                                kernel_max: int = 20,
                                intensity: float = 0.1,
                                validation_split: float = 0.1,
                                calib_split: float = 0.2,
                                test_split: float = 0.1,
                                transform = None,
                                normalization: Literal[None, 'standard', 'min-max'] = None,
                                num_workers: int = 4,
                                pin_memory: bool = True) -> tuple[DataLoader, DataLoader, np.ndarray, np.ndarray]:
    
    dataset = MotionBlurDataset(data_root=data_root, n_blur_samples=n_blur_samples, 
                                kernel_min=kernel_min, kernel_max=kernel_max, intensity=intensity, 
                                transform=transform, normalization=normalization)
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)

    datasize_size = len(dataset)
    validation_size = int(np.floor(validation_split * datasize_size))
    calib_size = int(np.floor(calib_split * datasize_size))
    test_size = int(np.floor(test_split * datasize_size))
    train_size = datasize_size - validation_size - calib_size - test_size

    train_indices = indices[:train_size]
    validation_indices = indices[train_size:train_size + validation_size]
    test_indices = indices[train_size + validation_size:train_size + validation_size + test_size]
    calib_indices = indices[train_size + validation_size + test_size:]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    validation_dataset = torch.utils.data.Subset(dataset, validation_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False,
                                   num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, validation_loader, calib_indices, test_indices

def get_motionblur_calib(calib_indices: np.ndarray,
                         data_root: str = '/share/monakhova/Cassandra_data/UQNet_proj/motionblur',
                         batch_size: int = 16,
                         n_blur_samples: int = 5,
                         kernel_min: int = 5,
                         kernel_max: int = 20,
                         intensity: float = 0.1,
                         transform = None,
                         normalization: Literal[None, 'standard', 'min-max'] = None,
                         num_workers: int = 4,
                         pin_memory: bool = True) -> DataLoader:
    kernels = np.linspace(kernel_min, kernel_max, n_blur_samples).astype(int).tolist()
    dataset = MotionBlurDataset(data_root=data_root, kernels=kernels, intensity=intensity, 
                                transform=transform, normalization=normalization)
    calib_dataset = torch.utils.data.Subset(dataset, calib_indices)
    calib_loader = DataLoader(calib_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    return calib_loader

def get_motionblur_test(test_indices: np.ndarray,
                        data_root: str = '/share/monakhova/Cassandra_data/UQNet_proj/motionblur',
                        batch_size: int = 16,
                        n_blur_samples: int = 5,
                        kernel_min: int = 5,
                        kernel_max: int = 20,
                        intensity: float = 0.1,
                        transform = None,
                        normalization: Literal[None, 'standard', 'min-max'] = None,
                        num_workers: int = 4,
                        pin_memory: bool = True) -> DataLoader:
    kernels = np.linspace(kernel_min, kernel_max, n_blur_samples).astype(int).tolist()
    dataset = MotionBlurDataset(data_root=data_root, kernels=kernels, intensity=intensity, 
                                transform=transform, normalization=normalization)
    test_dataset = torch.utils.data.Subset(dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)
    return test_loader

if __name__ == "__main__":
    dataset = MotionBlurDataset()
    train_loader, test_loader, calib_indices, test_indices = get_motionblur_dataloaders(
        data_root='/share/monakhova/Cassandra_data/UQNet_proj/motionblur',
        batch_size=16,
        n_blur_samples=5,
        kernel_min=5,
        kernel_max=20,
        intensity=0.1,
        validation_split=0.1,
        calib_split=0.2,
        test_split=0.1
    )
    print(f"Train loader size: {len(train_loader)}")
    print(f"Test loader size: {len(test_loader)}")
    print(f"Calib indices: {calib_indices}")
    print(f"Test indices: {test_indices}")