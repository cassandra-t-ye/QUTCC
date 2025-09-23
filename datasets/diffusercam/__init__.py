import os
from typing import Literal, Tuple

import cv2
import imageio.v3 as iio
import numpy as np
import pandas as pd
import torch
from numpy.fft import fft2, ifft2, ifftshift
from torch.utils.data import DataLoader, Dataset, Subset

from datasets.utils import normalize


class DiffuserDataset(Dataset):
    def __init__(self, data_root: str,
                 type: Literal['train', 'test'], transform=None, 
                 normalization: Literal[None, 'standard', 'min-max']=None):
        self.csv_path = data_root + "/diffusercam.csv"
        self.data_dir = data_root + "/diffuser_images"
        self.gt_dir = data_root + "/ground_truth_lensed"
        csv_name = "dataset_train.csv" if type == 'train' else "dataset_test.csv"
        self.paths = pd.read_csv(os.path.join(data_root, csv_name), header=None)

        before = len(self.paths)
        self.paths = self.paths[self.paths[0].apply(self.files_exist)].reset_index(drop=True)
        after = len(self.paths)

        if before != after:
            print(f"[DiffuserDataset] Skipped {before - after} entries "
                  "because image or ground truth file was missing")

        self.transform = transform
        self.normalization = normalization
        self.H_conj = np.conj(fft2(ifftshift(self.pad(load_psf(data_root + "/psf.tiff")), axes=(0, 1)), axes=(0, 1), norm='ortho'))
    
    def __len__(self):
        return len(self.paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        base_path = self.paths.iloc[idx, 0]
        image_path = self.data_dir + "/" + base_path
        gt_path = self.gt_dir + "/" + base_path
        image, gt = iio.imread(image_path), iio.imread(gt_path)
        image, gt = image / image.max(), gt / gt.max()
        image = cv2.resize(image, (512, 288), interpolation=cv2.INTER_CUBIC) 
        gt = cv2.resize(gt, (512, 288), interpolation=cv2.INTER_CUBIC)
        image, gt = rgb2gray(image), rgb2gray(gt)
        adjoint_image = self.adjoint(image)

        input_tensor = torch.tensor(adjoint_image, dtype=torch.float32).unsqueeze(0)
        gt_tensor = torch.tensor(gt, dtype=torch.float32).unsqueeze(0)

        if self.transform:
            input_tensor = self.transform(input_tensor)
    
        if self.normalization:
            input_tensor = normalize(input_tensor, type=self.normalization, per_pixel=False)
            gt_tensor = normalize(gt_tensor, type=self.normalization, per_pixel=False)
        
        return input_tensor, gt_tensor
    
    def adjoint(self, x):
        X = fft2(self.pad(x), axes=(0, 1), norm='ortho')
        return self.crop(np.real(ifft2(self.H_conj * X, axes=(0, 1), norm='ortho')))

    def pad(self, img: np.ndarray) -> np.ndarray:
        return np.pad(img, ((img.shape[0]//2, img.shape[0]//2), 
                            (img.shape[1]//2, img.shape[1]//2)), mode='constant')
    
    def crop(self, img):
        return img[img.shape[0]//4:-img.shape[0]//4, img.shape[1]//4: -img.shape[1]//4]
    
    def files_exist(self, rel_path: str) -> bool:
        img_ok = os.path.exists(os.path.join(self.data_dir, rel_path))
        gt_ok = os.path.exists(os.path.join(self.gt_dir, rel_path))
        return img_ok and gt_ok
    


def load_psf(path: str = "/share/monakhova/Cassandra_data/UQNet_proj/diffusercam/psf.tff") -> torch.Tensor:
    psf = iio.imread(path)
    psf = psf - np.mean(psf[0:15,0:15])
    psf = cv2.resize(psf, (512, 288), interpolation=cv2.INTER_CUBIC) 
    psf = rgb2gray(psf)
    psf = psf / np.linalg.norm(psf)
    return psf

def rgb2gray(rgb: np.ndarray) -> np.ndarray:
    return np.dot(rgb[...,:3], [0.2126, 0.7152, 0.0722])

def get_diffusercam_dataloaders(data_root: str = "/share/monakhova/Cassandra_data/UQNet_proj/diffusercam", 
                                batch_size: int = 16, 
                                validation_split: float = 0.1, 
                                calib_split: float = 0.2, 
                                transform = None, 
                                normalization: Literal[None, 'standard', 'min-max'] = None,
                                num_workers: int = 4,
                                pin_memory: bool = True) -> Tuple[DataLoader, DataLoader, np.ndarray]:
    dataset = DiffuserDataset(data_root, 'train', transform, normalization)
    indices = np.arange(len(dataset))
    np.random.shuffle(indices)
    
    dataset_size = len(indices)
    validation_size = int(np.floor(validation_split * dataset_size))
    calib_size = int(np.floor(calib_split * dataset_size))
    train_size = dataset_size - validation_size - calib_size
    train_indices = indices[:train_size]
    validation_indices = indices[train_size:train_size + validation_size]
    calib_indices = indices[train_size + validation_size:]

    train_dataset = Subset(dataset, train_indices)
    validation_dataset = Subset(dataset, validation_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin_memory)
    validation_loader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False,
                                   num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, validation_loader, calib_indices

def get_diffusercam_calib(calib_indices: np.ndarray,
                          data_root: str = "/share/monakhova/Cassandra_data/UQNet_proj/diffusercam", 
                          batch_size: int = 16, 
                          transform = None, 
                          normalization: Literal[None, 'standard', 'min-max'] = None,
                          num_workers: int = 4,
                          pin_memory: bool = True) -> DataLoader:
    dataset = DiffuserDataset(data_root, 'train', transform, normalization)
    calib_dataset = Subset(dataset, calib_indices)
    calib_loader = DataLoader(calib_dataset, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory)
    return calib_loader

def get_diffusercam_test(data_root: str = "/share/monakhova/Cassandra_data/UQNet_proj/diffusercam", 
                         batch_size: int = 16, 
                         transform = None, 
                         normalization: Literal[None, 'standard', 'min-max'] = None,
                         num_workers: int = 4,
                         pin_memory: bool = True) -> DataLoader:
    test_dataset = DiffuserDataset(data_root, 'test', transform, normalization)
    # 200 index subset
    test_indices = np.arange(200, 400)
    test_dataset = Subset(test_dataset, test_indices)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory)
    return test_loader

if __name__ == "__main__":
    data_root = "/share/monakhova/Cassandra_data/UQNet_proj/diffusercam"
    train_loader, validation_loader, calib_indices = get_diffusercam_dataloaders(data_root, num_workers=0)
    test_dataset = DiffuserDataset(data_root, 'test')
    print(len(train_loader.dataset), len(test_dataset))