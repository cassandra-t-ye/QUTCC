# Example command to evaluate MRI: python  -m calculate_statistics.calculate_statistics --task mri
import sys
import os
import argparse 
import glob
import yaml
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import logging
from models import create_model, create_model2, create_im2im
import torch.nn as nn
from datasets.fastmri import FastMRIDataset
import datasets.fastmri.utils as fastmri_utils
from torch.utils.data import TensorDataset, DataLoader
from datasets.fmd.data_loader import (load_noisy_single_loader,
                                                load_real_noise_loader)
from typing import Any, Callable, Literal
from datasets.diffusercam import get_diffusercam_calib, get_diffusercam_test
from datasets.fastmri import FastMRIDataset
from datasets.LidcIdri.LIDCLDRI import load_ct_dataloader

from models.checkpointing import load_checkpoint_for_inference
from evaluation import (
    check_quantile_crossings,
    get_test_intervals_missed,
    return_calibrated_bounds,
    size_stratified_risk,
)
import json
from datasets.bsccm import get_bsccm_calib, get_bsccm_test
from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics.image import PeakSignalNoiseRatio
import lpips

mse_loss = nn.MSELoss()

def get_test_dataloader(experiment_type, info, net=Literal['im2im', 'quantile'], batch_size_multiplier=0.5, test_subset=200):
    if experiment_type == "mri":
        mask_info = {'type': 'equispaced', 'center_fraction' : [0.08], 'acceleration' : [4]}
        test_dataset = FastMRIDataset(info["test_data_path"], normalize_input='standard', normalize_output='min-max', 
                                      mask_info=mask_info, num_volumes=20)
        fastmri_utils.normalize_dataset(test_dataset)
        test_dataloader = DataLoader(test_dataset, batch_size=int(info["batch_size"] * batch_size_multiplier), 
                                     shuffle=False, num_workers=1, pin_memory=False)
    
    elif experiment_type in ["gaussian", "poisson"]:
        noise_type, min_noise, max_noise, num_noise = experiment_type, info["min_noise"], info["max_noise"], info["num_noise"]
        test_dataloader = load_noisy_single_loader(noise_type, min_noise, max_noise, num_noise, 
                                                   image_folder=info["test_data_path"], 
                                                   batch_size=int(info["batch_size"] * batch_size_multiplier),
                                                   types=None, transform=None)
    
    elif experiment_type == "qpi":
        run_folder = info["experiments_folder"] + f"/{experiment_type}/{info[f'{net}_root']}"
        test_indices_path = run_folder + "/test_indices.json"
        with open(test_indices_path, "r") as f:
            test_indices = json.load(f)
        test_dataloader = get_bsccm_test(test_indices[:test_subset], info["data_root"], 
                                         batch_size=int(info["batch_size"] * batch_size_multiplier), transform=None, 
                                         normalization='min-max', num_workers=0, pin_memory=False)

    elif experiment_type == "real_noise_mice":
        gt_pathway = info["test_data_path"] + "/gt"
        noisy_pathway = info["test_data_path"] + "/noisy"
        test_dataloader = load_real_noise_loader(noisy_path=noisy_pathway, gt_path=gt_pathway, batch_size=int(info["batch_size"] * batch_size_multiplier))

    elif experiment_type == "CT":
        test_dataloader = load_ct_dataloader(root_path=info["test_data_path"], dataset_type="test", batch_size = info["batch_size"], steps=info["steps"])


    print(f"Test dataset size: {len(test_dataloader.dataset)}")
    return test_dataloader

if __name__ == "__main__":
    torch.manual_seed(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)


    parser = argparse.ArgumentParser(
        description="Run evaluation for a given 'task' (e.g. qpi, motionblur, etc.)."
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="Name of the experiment/task (matches keys in BEST_RUNS)."
    )
    args = parser.parse_args()
    task = args.task


    from analysis import BEST_RUNS
    im2im_model_desc = "im2im"
    im2im_model, run_folder = load_checkpoint_for_inference(
        net="im2im",
        in_channels=BEST_RUNS[task]["in_channels"],
        experiment_type=BEST_RUNS[task]["experiment_type"],
        run_folder_root=BEST_RUNS[task][f"{im2im_model_desc}_root"],
        epoch=BEST_RUNS[task][f"{im2im_model_desc}_epoch"],
        device=device,
    )

    quant_model, runfolder = load_checkpoint_for_inference(
        net="unet_quantile",
        in_channels=BEST_RUNS[task]["in_channels"],
        experiment_type=BEST_RUNS[task]["experiment_type"],
        run_folder_root=BEST_RUNS[task]["quantile_root"],
        epoch=BEST_RUNS[task]["quantile_epoch"],
        device=device,
    )

    unet_im2im_model, run_folder = load_checkpoint_for_inference(
        net="unet_im2im",
        in_channels=BEST_RUNS[task]["in_channels"],
        experiment_type=BEST_RUNS[task]["experiment_type"],
        run_folder_root=BEST_RUNS[task]["unet_im2im_root"],
        epoch=BEST_RUNS[task]["unet_im2im_epoch"],
        device=device,
    )

    batch_size = BEST_RUNS[task]["batch_size"]


    test_subset = 2000
    test_dataloader = get_test_dataloader(task, BEST_RUNS[task], net="quantile", batch_size_multiplier=0.7, test_subset=test_subset)

    dataloader = test_dataloader

    batch_mse = torch.nn.MSELoss(reduction="none")
    ssim_metric = StructuralSimilarityIndexMeasure(
        data_range=1.0, reduction="none"
    ).to(device)
    psnr_metric = PeakSignalNoiseRatio(
        data_range=1.0, reduction="none", dim=(1, 2, 3)
    ).to(device)
    lpips_metric = lpips.LPIPS(net="vgg").to(device)

    crossings_dataloader = dataloader


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


    with torch.no_grad():
        for batch in tqdm(dataloader):
            noisy, clean = batch
            clean_im2im = clean[:, 0, :, :].to(device)
            clean = clean.to(device)

            # ─── Im2Im ──────────────────────────────────────────────────────────────
            curr_denoised_lam = im2im_model(noisy.to(device))
            im2im_output = torch.unsqueeze(curr_denoised_lam[:, 1, :, :], 1)
            im2im_clean = clean.to(device)

            output = batch_mse(im2im_output, im2im_clean)
            output = output.view(im2im_output.size(0), -1).mean(dim=1)
            im2im_mse_acc.append(output)

            im2im_ssim_acc.append(
                ssim_metric(im2im_output, im2im_clean)
                .view(im2im_output.size(0), -1)
                .mean(dim=1)
            )
            im2im_psnr_acc.append(
                psnr_metric(im2im_output, im2im_clean)
                .view(im2im_output.size(0), -1)
                .mean(dim=1)
            )

            # reshaping for LPIPS calculation
            im2im_in  = (im2im_output.clamp(0,1) * 2.0 - 1.0)
            clean_in  = (im2im_clean.clamp(0,1) * 2.0 - 1.0)
            im2im_rgb = im2im_in.expand(-1, 3, -1, -1)
            clean_rgb = clean_in .expand(-1, 3, -1, -1)
            im2im_lpips_acc.append(lpips_metric(im2im_rgb, clean_rgb))

            # ─── Quantile ───────────────────────────────────────────────────────────
            q_prediction = torch.tensor([0.5], device=device, dtype=torch.float32)
            quantile_output = quant_model(noisy.to(device), q_prediction.to(device))
            quantile_clean = clean.to(device)

            output = batch_mse(quantile_clean, quantile_output)
            output = output.view(quantile_clean.size(0), -1).mean(dim=1)
            quantile_mse_acc.append(output)
            quantile_ssim_acc.append(
                ssim_metric(quantile_output, quantile_clean)
                .view(quantile_output.size(0), -1)
                .mean(dim=1)
            )
            quantile_psnr_acc.append(
                psnr_metric(quantile_output, quantile_clean)
                .view(quantile_output.size(0), -1)
                .mean(dim=1)
            )

            # reshaping for LPIPS calculation
            quantile_output  = (quantile_output.clamp(0,1) * 2.0 - 1.0)
            quantile_clean  = (quantile_clean.clamp(0,1) * 2.0 - 1.0)
            quantile_output = quantile_output.expand(-1, 3, -1, -1)
            quantile_clean = quantile_clean.expand(-1, 3, -1, -1)
            quantile_lpips_acc.append(lpips_metric(quantile_output, quantile_clean))

            # ─── Unet Im2Im ─────────────────────────────────────────────────────────
            timevect = torch.full(
                (noisy.shape[0],), 0.5, device=device, dtype=torch.float32
            )

            unet_im2im_output = unet_im2im_model(
                noisy.to(device), timevect.to(device)
            )
            unet_im2im_output = torch.unsqueeze(unet_im2im_output[:, 1, :, :], 1)
            unet_im2im_clean = clean.to(device)

            output = batch_mse(unet_im2im_clean, unet_im2im_output)
            output = output.view(unet_im2im_output.size(0), -1).mean(dim=1)
            unet_im2im_mse_acc.append(output)
            unet_im2im_ssim_acc.append(
                ssim_metric(unet_im2im_output, unet_im2im_clean)
                .view(unet_im2im_output.size(0), -1)
                .mean(dim=1)
            )
            unet_im2im_psnr_acc.append(
                psnr_metric(unet_im2im_output, unet_im2im_clean)
                .view(unet_im2im_output.size(0), -1)
                .mean(dim=1)
            )
            # reshaping for LPIPS calculation
            unet_im2im_output  = (unet_im2im_output.clamp(0,1) * 2.0 - 1.0)
            unet_im2im_clean  = (unet_im2im_clean.clamp(0,1) * 2.0 - 1.0)
            unet_im2im_output = unet_im2im_output.expand(-1, 3, -1, -1)
            unet_im2im_clean = unet_im2im_clean.expand(-1, 3, -1, -1)
            unet_im2im_lpips_acc.append(lpips_metric(unet_im2im_output, unet_im2im_clean))

    # ─── Concatenate all metrics ─────────────────────────────────────────────────────
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

    print(f"im2im_uq MSE: {im2im_mse_acc.mean().item():.7f} +/- {im2im_mse_acc.std().item():.7f}")
    print(f"im2im_uq SSIM: {im2im_ssim_acc.mean().item():.3f} +/- {im2im_ssim_acc.std().item():.3f}")
    print(f"im2im_uq PSNR: {im2im_psnr_acc.mean().item():.3f} +/- {im2im_psnr_acc.std().item():.3f}")
    print(f"im2im_uq LPIPS: {im2im_lpips_acc.mean().item():.3f} +/- {im2im_lpips_acc.std().item():.3f}")

    print(f"QUTCC MSE: {quantile_mse_acc.mean().item():.7f} +/- {quantile_mse_acc.std().item():.7f}")
    print(f"QUTCC SSIM: {quantile_ssim_acc.mean().item():.3f} +/- {quantile_ssim_acc.std().item():.3f}")
    print(f"QUTCC PSNR: {quantile_psnr_acc.mean().item():.3f} +/- {quantile_psnr_acc.std().item():.3f}")
    print(f"QUTCC LPIPS: {quantile_lpips_acc.mean().item():.3f} +/- {quantile_lpips_acc.std().item():.3f}")

    print(
        f"im2im_deep MSE: {unet_im2im_mse_acc.mean().item():.7f} +/- {unet_im2im_mse_acc.std().item():.7f}"
    )
    print(
        f"im2im_deep SSIM: {unet_im2im_ssim_acc.mean().item():.3f} +/- {unet_im2im_ssim_acc.std().item():.3f}"
    )
    print(
        f"im2im_deep PSNR: {unet_im2im_psnr_acc.mean().item():.3f} +/- {unet_im2im_psnr_acc.std().item():.3f}"
    )
    print(
        f"im2im_deep LPIPS: {unet_im2im_lpips_acc.mean().item():.3f} +/- {unet_im2im_lpips_acc.std().item():.3f}"
    )
