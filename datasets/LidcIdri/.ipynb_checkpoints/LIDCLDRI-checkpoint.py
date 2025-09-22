import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from typing import Dict, Tuple
from scipy.interpolate import interp1d
import scipy.fftpack as fft
from skimage.transform import rotate
from deepinv.datasets import LidcIdriSliceDataset
import torchvision.transforms as transforms
import os
import pickle
from . import backproject

class CT_Inverse_Cached(Dataset):
    def __init__(self, gt_dataset, step_size=800, cache_dir=None):
        self.gt_dataset = gt_dataset
        self.step_size = step_size
        self.cache_dir = cache_dir
        self._cache = {}
        
        # Create cache directory if specified
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
            self.cache_file = os.path.join(cache_dir, f'ct_cache_steps_{step_size}.pkl')
            self._load_cache()
    
    def _load_cache(self):
        """Load existing cache from disk"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, 'rb') as f:
                    self._cache = pickle.load(f)
                print(f"Loaded cache with {len(self._cache)} items")
            except:
                print("Failed to load cache, starting fresh")
                self._cache = {}
    
    def _save_cache(self):
        """Save cache to disk"""
        if self.cache_dir:
            with open(self.cache_file, 'wb') as f:
                pickle.dump(self._cache, f)
    
    def __len__(self):
        return len(self.gt_dataset)
    
    def __getitem__(self, idx):
        """
        Returns a tuple of (noisy_image, gt_image)
        """
        # Check if this item is already cached
        if idx in self._cache:
            return self._cache[idx]
        
        gt_image = self.gt_dataset[idx]
        
        # Create noisy version using radon transform and backprojection
        radon_test = backproject.radon(gt_image, self.step_size)
        noisy = torch.tensor(backproject.back_project(radon_test), dtype=torch.float32)
        noisy = torch.clamp(noisy / torch.max(noisy), 0, 1)
        
        if not isinstance(gt_image, torch.Tensor):
            gt_image = torch.tensor(gt_image, dtype=torch.float32)
        
        result = (noisy.unsqueeze(0), gt_image.unsqueeze(0))
        
        # Cache the result
        self._cache[idx] = result
        
        # Periodically save cache
        if len(self._cache) % 100 == 0:
            self._save_cache()
        
        return result
    
    def __del__(self):
        """Save cache when object is destroyed"""
        if hasattr(self, '_cache') and self.cache_dir:
            self._save_cache()

class CT_Inverse_Precomputed(Dataset):
    def __init__(self, gt_dataset, step_size=800, precompute_all=True):
        self.gt_dataset = gt_dataset
        self.step_size = step_size
        self.precomputed_data = []
        
        if precompute_all:
            self._precompute_all()
    
    def _precompute_all(self):
        """Precompute all radon/backprojection pairs at initialization"""
        print(f"Precomputing {len(self.gt_dataset)} samples...")
        
        for idx in tqdm(range(len(self.gt_dataset))):
            gt_image = self.gt_dataset[idx]
            
            # Create noisy version using radon transform and backprojection
            radon_test = backproject.radon(gt_image, self.step_size)
            noisy = torch.tensor(backproject.back_project(radon_test), dtype=torch.float32)
            noisy = torch.clamp(noisy / torch.max(noisy), 0, 1)
            
            if not isinstance(gt_image, torch.Tensor):
                gt_image = torch.tensor(gt_image, dtype=torch.float32)
            
            self.precomputed_data.append((noisy.unsqueeze(0), gt_image.unsqueeze(0)))
        
        print("Precomputation complete!")
    
    def __len__(self):
        return len(self.gt_dataset)
    
    def __getitem__(self, idx):
        return self.precomputed_data[idx]

def safe_max_normalize(x):
    """
    Safe max normalization that handles edge cases.
    Works with raw pixel values (not HU units).
    """
    x = x.astype(np.float32)
    normalized = x / np.max(x)
    return np.clip(normalized.astype(np.float32), 0, 1)

def load_dataset(root_path, dataset_type):
    preprocessing = transforms.Compose([transforms.Lambda(lambda x: safe_max_normalize(x))])
    dataset = LidcIdriSliceDataset(
        root=root_path,
        hounsfield_units=False,
        transform=preprocessing
    )
    
    train_size = int(0.6 * len(dataset))
    test_size = int(0.1 * len(dataset))
    calib_size = int(0.2 * len(dataset))
    valid_size = len(dataset) - train_size - calib_size - test_size
    
    train_dataset, test_dataset, calib_dataset, valid_dataset = random_split(
        dataset, [train_size, test_size, calib_size, valid_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    if dataset_type == "train":
        return train_dataset
    if dataset_type == "test":
        return test_dataset
    if dataset_type == "calibrate":
        return calib_dataset
    if dataset_type == "validate":
        return valid_dataset
    
    print("ERROR: NO DATASET RETURNED")
    return None

def load_ct_dataloader(root_path, dataset_type, batch_size, steps, use_cache=True):
    base_dataset = load_dataset(root_path, dataset_type)
    
    if use_cache:
        ct_dataset = CT_Inverse_Cached(base_dataset, step_size=steps, cache_dir=f"./cache_{dataset_type}")
    else:
        # Use precomputed version for smaller datasets
        ct_dataset = CT_Inverse_Precomputed(base_dataset, step_size=steps)
    
    dataloader = DataLoader(
        ct_dataset, 
        batch_size=batch_size, 
        num_workers=4  # Increased num_workers for better performance
    )
    return dataloader