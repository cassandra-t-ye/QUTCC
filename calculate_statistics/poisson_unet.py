import sys
import os

# get current working directory (where your notebook or script is running)
cwd = os.getcwd()

# compute its parent directory
parent = os.path.abspath(os.path.join(cwd, os.pardir))

# insert it at the front of sys.path
if parent not in sys.path:
    sys.path.insert(0, parent)

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

torch.manual_seed(1)

from datasets.denoising_fmd.data_loader import load_noisy_single_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

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

mod_type = "im2im"
path = '/share/monakhova/Cassandra_data/UQNet_proj/experiments/poisson/im2im_inchannels1_05_09/checkpoints/epoch50.pth'
config_path = '../models/model_config.yaml'
device = device
im2im_model, timevect = load_model(mod_type, path, config_path, device)

path = '/share/monakhova/Cassandra_data/UQNet_proj/experiments/poisson/unet_epochs100_bs4_2025-05-12-08.36.06/checkpoints/epoch50.pth'

quant_config = '../models/model_config.yaml'
#finetuned_quantile_path = '/home/ty392/UQNet/code_for_quant/model_weights/gaussian_quantile_95.pth'
quant_model, timevect = load_model(model_type = 'quantile', model_path = path, config_path = quant_config, device = "cuda")

#Validation DATASET:
validation_data_path = "/share/monakhova/Cassandra_data/UQNet_proj/Denoising_FMD/validation"


noise_type, min_noise, max_noise, num_noise = 'poisson', 80, 100, 20


val_dataloader = load_noisy_single_loader(noise_type, min_noise, max_noise, num_noise, image_folder=validation_data_path, batch_size=16,  types=None, transform=None)
val_dataset = val_dataloader.dataset





from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio
import lpips 

dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

batch_mse = torch.nn.MSELoss(reduction='none')
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0, reduction='none').to(device)
psnr_metric = PeakSignalNoiseRatio(data_range=1.0, reduction='none', dim=(1,2,3)).to(device)
lpips_metric = lpips.LPIPS(net='vgg').to(device)



quantile_mse_acc = []
quantile_ssim_acc = []
quantile_psnr_acc = []
quantile_lpips_acc = []


with torch.no_grad():
    for batch in tqdm(dataloader):
        noisy, clean = batch

        clean = clean.to(device)

        # For Quantile
        q_prediction = torch.tensor([0.5], device=device, dtype=torch.float32)
        quantile_output = quant_model(noisy.to(device), q_prediction.to(device))
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

print(f"Quantile MSE: {quantile_mse_acc.mean().item():.7f} +/- {quantile_mse_acc.std().item():.7f}")
print(f"Quantile SSIM: {quantile_ssim_acc.mean().item():.3f} +/- {quantile_ssim_acc.std().item():.3f}")
print(f"Quantile PSNR: {quantile_psnr_acc.mean().item():.3f} +/- {quantile_psnr_acc.std().item():.3f}")
print(f"Quantile LPIPS: {quantile_lpips_acc.mean().item():.3f} +/- {quantile_lpips_acc.std().item():.3f}")



















from evaluation import (check_quantile_crossings, get_test_intervals_missed,
                     return_calibrated_bounds,
                        size_stratified_risk)
from evaluation.calibration import (compute_optimal_lambdas,
                                            compute_optimal_lambdas_quantile)

calib_data_path = '/share/monakhova/Cassandra_data/UQNet_proj/Denoising_FMD/calibration'
noise_type, min_noise, max_noise, num_noise = 'poisson', 0, 25, 20
calib_dataloader = load_noisy_single_loader(noise_type, min_noise, max_noise, num_noise, 
                                            image_folder=calib_data_path, batch_size=4, types=None, transform=None)
calib_dataset = calib_dataloader.dataset
print(f"Calibration dataset size: {len(calib_dataset)}")

alpha, min_l, max_l, num_l = 0.15, 0, 3, 100
im2im_lam = compute_optimal_lambdas(dataset=calib_dataset, model=im2im_model, alpha=alpha, 
                                    min_lam=min_l, max_lam=max_l, num_lam=num_l, device=device)

target_alpha = 0.1 / 2
lower_min, lower_max = 1e-10, 0.1
upper_min, upper_max = 0.99, 1- 1e-10
lower_q, upper_q = compute_optimal_lambdas_quantile(dataloader=calib_dataloader, model=quant_model, alpha=target_alpha, 
                                                    lower_min=lower_min, lower_max=lower_max, 
                                                    upper_min=upper_min, upper_max=upper_max,
                                                    max_iterations=20, device=device)



test_data_path = "/share/monakhova/Cassandra_data/UQNet_proj/Denoising_FMD/validation"
noise_type, min_noise, max_noise, num_noise = 'poisson', 80, 100, 20
test_dataloader = load_noisy_single_loader(noise_type, min_noise, max_noise, num_noise, 
                                          image_folder=test_data_path, batch_size=4, types=None, transform=None)
print(f"Test dataset size: {len(test_dataloader.dataset)}")
test_dataset = test_dataloader.dataset

df_intervals_missed, rows, totals = get_test_intervals_missed(test_dataloader=test_dataloader,
                                                              im2im_model=im2im_model, im2im_lam=im2im_lam,
                                                              quantile_model=quant_model, lower_q=lower_q, upper_q=upper_q,
                                                              device=device)

percent_missed = {m: v["missed"] / v["n"] for m, v in totals.items()}
print({k: f"{v:.4f}" for k, v in percent_missed.items()})


im2im_df = df_intervals_missed[df_intervals_missed["method"] == "im2im"][["interval", "missed"]]
quantile_df = df_intervals_missed[df_intervals_missed["method"] == "quantile"][["interval", "missed"]]

im2im_stratified = size_stratified_risk(im2im_df)
quantile_stratified = size_stratified_risk(quantile_df)