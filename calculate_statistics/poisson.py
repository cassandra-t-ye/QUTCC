import os
import sys

# get current working directory (where your notebook or script is running)
cwd = os.getcwd()

# compute its parent directory
parent = os.path.abspath(os.path.join(cwd, os.pardir))

# insert it at the front of sys.path
if parent not in sys.path:
    sys.path.insert(0, parent)

import glob
import logging
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from models import create_im2im, create_model, create_model2

mse_loss = nn.MSELoss()
import pickle
import random
from collections import defaultdict

import imageio
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns
from torch.utils.data import DataLoader, TensorDataset

import datasets.fastmri.utils as fastmri_utils
from datasets.fastmri import FastMRIDataset

torch.manual_seed(1)

from datasets.denoising_fmd.data_loader import load_noisy_single_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

from analysis import BEST_RUNS
from evaluation import (check_quantile_crossings, get_test_intervals_missed,
                        return_calibrated_bounds, size_stratified_risk)
from models.checkpointing import load_checkpoint_for_inference

task = "poisson"
im2im_model, run_folder = load_checkpoint_for_inference(net='im2im', in_channels=1, experiment_type=task, 
                              run_folder_root=BEST_RUNS[task]["im2im_root"], epoch=BEST_RUNS[task]["im2im_epoch"], device=device)

quant_model, run_folder = load_checkpoint_for_inference(net='unet_quantile', in_channels=1, experiment_type=task, 
                              run_folder_root=BEST_RUNS[task]["quantile_root"], epoch=BEST_RUNS[task]["quantile_epoch"], device=device)

unet_im2im_model, run_folder = load_checkpoint_for_inference(net='unet_im2im', in_channels=1, experiment_type=task, 
                              run_folder_root=BEST_RUNS[task]["unet_im2im_root"], epoch=BEST_RUNS[task]["unet_im2im_epoch"], device=device)

unet_model, run_folder = load_checkpoint_for_inference(net='unet', in_channels=1, experiment_type=task, 
                              run_folder_root=BEST_RUNS[task]["unet_root"], epoch=BEST_RUNS[task]["unet_epoch"], device=device)


batch_size = BEST_RUNS[task]["batch_size"]
timevect = torch.full((batch_size,), 0.5, device=device, dtype=torch.float32)


#Validation DATASET:
validation_data_path = "/share/monakhova/Cassandra_data/UQNet_proj/Denoising_FMD/validation"


noise_type, min_noise, max_noise, num_noise = 'poisson', 80, 100, 20


val_dataloader = load_noisy_single_loader(noise_type, min_noise, max_noise, num_noise, image_folder=validation_data_path, batch_size=batch_size,  types=None, transform=None)
val_dataset = val_dataloader.dataset





# from evaluation import (check_quantile_crossings, get_test_intervals_missed,
#                         load_checkpoint, return_calibrated_bounds,
#                         size_stratified_risk)
# from evaluation.calibration import (compute_optimal_lambdas,
#                                             compute_optimal_lambdas_quantile)
# crossings_dataloader = dataloader=(DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0))


# # Quantile results
# quantiles = torch.tensor([0.05, 0.95]).to(device)
# # quantiles = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]).to(device)
# crossings_data = check_quantile_crossings(dataloader=(DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)), model=quant_model, quantiles=quantiles, device=device)
# violations, total = crossings_data["violations"], crossings_data["total"]
# print(f"Violations: {violations}, Total: {total}")

# # im2im results
# def check_im2im_quantile_crossings(dataloader: DataLoader, model: nn.Module, 
#                              quantiles: torch.Tensor, device: str):
#     Q = len(quantiles)
#     violations, total = 0, 0

#     with torch.no_grad():
#         model.eval()
#         for noisy, _ in tqdm(dataloader, desc="Checking im2im Crossings"):
#             noisy: torch.Tensor = noisy.to(device)
#             # B, C, H, W = noisy.shape
#             print(noisy.shape)

#             # noisy = noisy.unsqueeze(0).expand(Q, -1, -1, -1, -1).reshape(-1, C, H, W)   # [Q * B, C, H, W]
#             # q = quantiles.unsqueeze(1).expand(-1, B).reshape(-1)                        # [Q * B]
#             # print(noisy.shape)
            
#             pred: torch.Tensor = model(noisy)                                        # [Q * B, 1, H, W]
#             # print(pred.shape)

#             violations += (pred[:, 0, :, :] > pred[:, 2, :, :]).sum().item()
#             total += torch.numel(pred[:, 0, :, :])

#     print(f"Total im2im crossing violations: {violations}")
#     print(f"Percentage of violations: {violations / total * 100}%")
#     return violations, total

# quantiles = torch.tensor([0.05, 0.95])
# violations, total = check_im2im_quantile_crossings(dataloader=crossings_dataloader, model=im2im_model, quantiles=quantiles, device=device)
# print(f"Violations: {violations}, Total: {total}")










import lpips
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio

dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

batch_mse = torch.nn.MSELoss(reduction='none')
ssim_metric = StructuralSimilarityIndexMeasure(data_range=1.0, reduction='none').to(device)
psnr_metric = PeakSignalNoiseRatio(data_range=1.0, reduction='none', dim=(1,2,3)).to(device)
lpips_metric = lpips.LPIPS(net='vgg').to(device)


im2im_mse_acc = []
im2im_ssim_acc = []
im2im_psnr_acc = []
im2im_lpips_acc = []

quantile_mse_acc = []
quantile_ssim_acc = []
quantile_psnr_acc = []
quantile_lpips_acc = []

unet_im2im_mse_acc = []
unet_im2im_ssim_acc = []
unet_im2im_psnr_acc = []
unet_im2im_lpips_acc = []

unet_mse_acc = []
unet_ssim_acc = []
unet_psnr_acc = []
unet_lpips_acc = []

with torch.no_grad():
    for batch in tqdm(dataloader):
        noisy, clean = batch
        clean_im2im = clean[:, 0, :, :].to(device)
        clean = clean.to(device)

        #For Im2Im:
        curr_denoised_lam = im2im_model(noisy.to(device))
        im2im_output = torch.unsqueeze(curr_denoised_lam[:, 1, :, :], 1)
        im2im_clean = clean.to(device)

        output = batch_mse(im2im_output, im2im_clean)
        output = output.view(im2im_output.size(0), -1).mean(dim=1)
        im2im_mse_acc.append(output)

        im2im_ssim_acc.append(ssim_metric(im2im_output, im2im_clean).view(im2im_output.size(0), -1).mean(dim=1))
        im2im_psnr_acc.append(psnr_metric(im2im_output, im2im_clean).view(im2im_output.size(0), -1).mean(dim=1))
        im2im_lpips_acc.append(lpips_metric(im2im_output, im2im_clean))

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

        # For Unet Im2Im

        timevect = torch.full((noisy.shape[0],), 0.5, device=device, dtype=torch.float32)

        unet_im2im_output = unet_im2im_model(noisy.to(device), timevect.to(device))
        unet_im2im_output = torch.unsqueeze(unet_im2im_output[:, 1, :, :], 1)
        unet_im2im_clean = clean.to(device)
        output = batch_mse(unet_im2im_clean, unet_im2im_output)
        output = output.view(unet_im2im_output.size(0), -1).mean(dim=1)
        unet_im2im_mse_acc.append(output)
        unet_im2im_ssim_acc.append(ssim_metric(unet_im2im_output, unet_im2im_clean).view(unet_im2im_output.size(0), -1).mean(dim=1))
        unet_im2im_psnr_acc.append(psnr_metric(unet_im2im_output, unet_im2im_clean).view(unet_im2im_output.size(0), -1).mean(dim=1))
        unet_im2im_lpips_acc.append(lpips_metric(unet_im2im_output, unet_im2im_clean))

        # For Unet
        unet_output = unet_model(noisy.to(device), timevect.to(device))
        unet_clean = clean.to(device)
        output = batch_mse(unet_clean, unet_output)
        output = output.view(unet_output.size(0), -1).mean(dim=1)
        unet_mse_acc.append(output)
        unet_ssim_acc.append(ssim_metric(unet_output, unet_clean).view(unet_output.size(0), -1).mean(dim=1))
        unet_psnr_acc.append(psnr_metric(unet_output, unet_clean).view(unet_output.size(0), -1).mean(dim=1))
        unet_lpips_acc.append(lpips_metric(unet_output, unet_clean))


im2im_mse_acc = torch.cat(im2im_mse_acc, dim=0)
im2im_ssim_acc = torch.cat(im2im_ssim_acc, dim=0)
im2im_psnr_acc = torch.cat(im2im_psnr_acc, dim=0)
im2im_lpips_acc = torch.cat(im2im_lpips_acc, dim=0)

quantile_mse_acc = torch.cat(quantile_mse_acc, dim=0)
quantile_ssim_acc = torch.cat(quantile_ssim_acc, dim=0)
quantile_psnr_acc = torch.cat(quantile_psnr_acc, dim=0)
quantile_lpips_acc = torch.cat(quantile_lpips_acc, dim=0)

unet_im2im_mse_acc = torch.cat(unet_im2im_mse_acc, dim=0)
unet_im2im_ssim_acc = torch.cat(unet_im2im_ssim_acc, dim=0)
unet_im2im_psnr_acc = torch.cat(unet_im2im_psnr_acc, dim=0)
unet_im2im_lpips_acc = torch.cat(unet_im2im_lpips_acc, dim=0)

unet_mse_acc = torch.cat(unet_mse_acc, dim=0)
unet_ssim_acc = torch.cat(unet_ssim_acc, dim=0)
unet_psnr_acc = torch.cat(unet_psnr_acc, dim=0)
unet_lpips_acc = torch.cat(unet_lpips_acc, dim=0)

print(f"im2im MSE: {im2im_mse_acc.mean().item():.7f} +/- {im2im_mse_acc.std().item():.7f}")
print(f"im2im SSIM: {im2im_ssim_acc.mean().item():.3f} +/- {im2im_ssim_acc.std().item():.3f}")
print(f"im2im PSNR: {im2im_psnr_acc.mean().item():.3f} +/- {im2im_psnr_acc.std().item():.3f}")
print(f"im2im LPIPS: {im2im_lpips_acc.mean().item():.3f} +/- {im2im_lpips_acc.std().item():.3f}")

print(f"Quantile MSE: {quantile_mse_acc.mean().item():.7f} +/- {quantile_mse_acc.std().item():.7f}")
print(f"Quantile SSIM: {quantile_ssim_acc.mean().item():.3f} +/- {quantile_ssim_acc.std().item():.3f}")
print(f"Quantile PSNR: {quantile_psnr_acc.mean().item():.3f} +/- {quantile_psnr_acc.std().item():.3f}")
print(f"Quantile LPIPS: {quantile_lpips_acc.mean().item():.3f} +/- {quantile_lpips_acc.std().item():.3f}")

print(f"im2im unet MSE: {unet_im2im_mse_acc.mean().item():.7f} +/- {unet_im2im_mse_acc.std().item():.7f}")
print(f"im2im unet SSIM: {unet_im2im_ssim_acc.mean().item():.3f} +/- {unet_im2im_ssim_acc.std().item():.3f}")
print(f"im2im unet PSNR: {unet_im2im_psnr_acc.mean().item():.3f} +/- {unet_im2im_psnr_acc.std().item():.3f}")
print(f"im2im unet LPIPS: {unet_im2im_lpips_acc.mean().item():.3f} +/- {unet_im2im_lpips_acc.std().item():.3f}")

print(f"Unet MSE: {unet_mse_acc.mean().item():.7f} +/- {unet_mse_acc.std().item():.7f}")
print(f"Unet SSIM: {unet_ssim_acc.mean().item():.3f} +/- {unet_ssim_acc.std().item():.3f}")
print(f"Unet PSNR: {unet_psnr_acc.mean().item():.3f} +/- {unet_psnr_acc.std().item():.3f}")
print(f"Unet LPIPS: {unet_lpips_acc.mean().item():.3f} +/- {unet_lpips_acc.std().item():.3f}")
