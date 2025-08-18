import sys
import os

cwd = os.getcwd()
parent = os.path.abspath(os.path.join(cwd, os.pardir))
if parent not in sys.path:
    sys.path.insert(0, parent)

import os
import glob
import yaml
import cv2
import numpy as np
import torch
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from models import create_model, create_model2, create_im2im
import torch.nn as nn
from tqdm import tqdm
mse_loss = nn.MSELoss()
import imageio
import pickle
import matplotlib.ticker as ticker
from datasets.fastmri import FastMRIDataset
import datasets.fastmri.utils as fastmri_utils
from torch.utils.data import TensorDataset, DataLoader
from collections import defaultdict
import pandas as pd
import random
import seaborn as sns

from datasets.denoising_fmd.data_loader import load_noisy_single_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

SEED = 1
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
# torch.cuda.manual_seed_all(SEED)
# torch.use_deterministic_algorithms(True)
# torch.backends.cudnn.benchmark = False
# torch.backends.cudnn.deterministic = True


def load_img(img_path: str):
    img = cv2.imread(img_path, -1)
    
    img_tensor = torch.from_numpy(img).clone().detach() 
    img_tensor = img_tensor.to(torch.float32).div(255.0)
    
    return torch.clamp(img_tensor, 0, 1)   

def load_yaml(file_path: str) -> dict:
    """Loads a YAML configuration file and returns the content as a dictionary."""
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def load_model(model_type: str, model_path: str, config_path: str, device: str):
    """
    Loads and initializes a U-Net model.

    Args:
        model_type (str): "diffusion", "quantile", "im2im".
        model_path (str): Path to the model checkpoint.
        config_path (str): Path to the YAML config file.
        device (str): Device to load the model onto ("cuda" or "cpu").

    Returns:
        tuple: (model, timevect)
    """
    # Load model configuration
    model_config = load_yaml(config_path)

    # Load and initialize the specified model
    print(torch.load(model_path, map_location=device, weights_only = False).keys())
    state_dict = torch.load(model_path, map_location=device, weights_only = False)["model_state_dict"]
    
    if model_type == "diffusion":
        model = create_model(**model_config, num_measurements=1).to(device)
    elif model_type == "quantile":
        model = create_model2(**model_config, num_measurements=1).to(device)
    elif model_type == "im2im":
        model = create_im2im(**model_config, num_measurements=1).to(device)
    else:
        raise ValueError("Invalid model type. Choose 'diffusion' or 'quantile'.")

    model.load_state_dict(state_dict)

    # Define time vector
    timevect = torch.tensor([0.5], device=device)
    print("Model Loaded!")
    return model, timevect

# path = '/home/ttk22/code_for_quant/models/mri_results/mri_quantile_10.pth'
path = '/share/monakhova/Cassandra_data/UQNet_proj/experiments/mri/unet_epochs50_bs12_2025-05-09-19.41.57/checkpoints/epoch10.pth'
quant_config = '../models/model_config.yaml'
#finetuned_quantile_path = '/home/ty392/UQNet/code_for_quant/model_weights/gaussian_quantile_95.pth'
quant_model, timevect = load_model(model_type = 'quantile', model_path = path, config_path = quant_config, device = "cuda")

#Validation DATASET:
validation_data_path = "/share/monakhova/Cassandra_data/UQNet_proj/Fast_MRI/validation"
mask_info = {'type': 'equispaced', 'center_fraction' : [0.08], 'acceleration' : [4]}
val_dataset = FastMRIDataset(validation_data_path, normalize_input='standard', normalize_output = 'min-max', mask_info=mask_info, num_volumes=4)
fastmri_utils.normalize_dataset(val_dataset)

from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio
import lpips 

dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=0)

batch_mse = torch.nn.MSELoss(reduction='none')
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0, reduction='none').to(device)
psnr_metric = PeakSignalNoiseRatio(data_range=1.0, reduction='none', dim=(1,2,3)).to(device)
lpips_metric = lpips.LPIPS(net='vgg').to(device)

quantile_mse_acc = []
quantile_ssim_acc = []
quantile_psnr_acc = []
quantile_lpips_acc = []


def count_trainable_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


with torch.no_grad():
    for batch in tqdm(dataloader):
        noisy, clean = batch

        clean = clean.to(device)
        # For Quantile
        q_prediction = torch.tensor([0.5], device=device, dtype=torch.float32)
        quantile_output = quant_model(noisy.to(device), q_prediction.to(device))
        print(count_trainable_params(quant_model))
        quantile_clean = clean.to(device)

        output = batch_mse(quantile_clean, quantile_output)
        output = output.view(quantile_clean.size(0), -1).mean(dim=1)
        quantile_mse_acc.append(output)
        quantile_ssim_acc.append(ssim_metric(quantile_output, quantile_clean).view(quantile_output.size(0), -1).mean(dim=1))
        quantile_psnr_acc.append(psnr_metric(quantile_output, quantile_clean).view(quantile_output.size(0), -1).mean(dim=1))
        quantile_lpips_acc.append(lpips_metric(quantile_output, quantile_clean))
quantile_mse_acc = torch.cat(quantile_mse_acc, dim=0)
quantile_ssim_acc = torch.cat(quantile_ssim_acc, dim=0)
quantile_psnr_acc = torch.cat(quantile_psnr_acc, dim=0)
quantile_lpips_acc = torch.cat(quantile_lpips_acc, dim=0)


print(f"Quantile MSE: {quantile_mse_acc.mean().item():.3f} +/- {quantile_mse_acc.std().item():.3f}")
print(f"Quantile SSIM: {quantile_ssim_acc.mean().item():.3f} +/- {quantile_ssim_acc.std().item():.3f}")
print(f"Quantile PSNR: {quantile_psnr_acc.mean().item():.3f} +/- {quantile_psnr_acc.std().item():.3f}")
print(f"Quantile LPIPS: {quantile_lpips_acc.mean().item():.3f} +/- {quantile_lpips_acc.std().item():.3f}")




