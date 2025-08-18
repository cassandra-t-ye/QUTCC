import gc
from collections import defaultdict
from typing import Dict, List, Literal

import numpy as np
import pandas as pd
import torch
from lpips import LPIPS
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure
from tqdm import tqdm

from models.quantile_uqnet import UNetModel
from models.im2im.add_uncertainty_im2im import ModelWithUncertainty
from models.im2im import Im2Im
from train import create_model, load_model


def get_binary_missed(lower_bound, upper_bound, gt):
    missed = (gt < lower_bound) | (gt > upper_bound)
    return missed.to(lower_bound.dtype)

def get_lower_upper_risk(lower, upper, gt):
    """Calculate the fraction of missed predictions based on quantile bounds.
    
    Args:
        lower: Lower bound predictions
        upper: Upper bound predictions
        label: Ground truth labels
    
    Returns:
        Int, Int: Fraction of missed low and high predictions
    """
    B, C, H, W = lower.shape
    total_pixels = B * H * W
    missed_low_cnt = (lower > gt).float().sum().item()
    missed_high_cnt = (upper < gt).float().sum().item()

    return missed_low_cnt / total_pixels, missed_high_cnt / total_pixels


def size_stratified_risk(df_intervals_missed: pd.DataFrame) -> Dict[str, float]:
    """
    Compute the average miss‑rate (“risk”) in four quartiles of interval length.

    Parameters
    ----------
    df_intervals_missed : pd.DataFrame
        Must contain
        * ``interval`` – numeric interval sizes
        * ``missed``   – one‑hot 0/1 (or bool) flag indicating a miss

    Returns
    -------
    Dict[str, float]
        Keys are the human‑friendly quartile labels
        ["Short", "Short-Medium", "Medium-Long", "Long"].
        Values are the mean miss‑rate in each length bin.
    """
    labels = ["Short", "Short-Medium", "Medium-Long", "Long"]
    df = df_intervals_missed.copy()
    df["interval"] = pd.to_numeric(df["interval"], errors="coerce")
    df["missed"]   = df["missed"].astype(float)
    df = df.dropna(subset=["interval", "missed"])
    df["bin"] = pd.qcut(df["interval"], len(labels), labels=labels)
    risk = df.groupby("bin", observed=True)["missed"].mean()
    return {label: risk.get(label, 0) for label in labels}

def return_calibrated_bounds(denoised_image, lam1):
    lower = denoised_image[:, 1, :, :] - lam1 * (denoised_image[:, 1, :, :] - denoised_image[:, 0, :, :])
    upper = lam1 * (denoised_image[:, 2, :, :] - denoised_image[:, 1, :, :]) + denoised_image[:, 1, :, :]
    return lower, upper

def load_checkpoint(net: Literal["im2im", "unet_quantile", "unet", "unet_im2im"], in_channels, 
                    experiment_type: Literal["mri", "gaussian", "poisson", "qpi", "lensless", "real_noise"],
                    run_folder_root: str,
                    epoch: int,
                    device: str = "cuda",
                    experiments_folder = "/share/monakhova/Cassandra_data/UQNet_proj/experiments"):
    run_folder = experiments_folder + f"/{experiment_type}/{run_folder_root}"
    ckpt_path = run_folder + f"/checkpoints/epoch{epoch}.pth"
    model = create_model(net=net, device=device, in_channels=in_channels, out_channels=1, debug=False)
    model = load_model(model, ckpt_path, device=device)

    return model, run_folder

def check_quantile_crossings(dataloader: DataLoader, model: nn.Module, 
                             quantiles: torch.Tensor, device: str):
    Q = len(quantiles)
    violations, total = 0, 0

    with torch.no_grad():
        model.eval()
        for noisy, _ in tqdm(dataloader, desc="Checking Quantile Crossings"):
            noisy: torch.Tensor = noisy.to(device)
            B, C, H, W = noisy.shape

            noisy = noisy.unsqueeze(0).expand(Q, -1, -1, -1, -1).reshape(-1, C, H, W)   # [Q * B, C, H, W]
            q = quantiles.unsqueeze(1).expand(-1, B).reshape(-1)                        # [Q * B]
            pred: torch.Tensor = model(noisy, q)                                        # [Q * B, 1, H, W]
            pred = pred.view(Q, B, 1, H, W).squeeze(2)                                  # [Q, B, H, W]

            violations += (pred[:-1] > pred[1:]).sum().item()
            total += (Q - 1) * B * H * W

    print(f"Total quantile crossing violations: {violations}")
    print(f"Percentage of violations: {violations / total * 100:.2f}%")

    del noisy, pred, q
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    return {"violations": violations, "total": total}

def get_risk(dataloader: DataLoader, im2im_model: ModelWithUncertainty, im2im_lambda: float,
             im2im_deep_model: ModelWithUncertainty, im2im_deep_lambda: float,
             quantile_model: UNetModel, lower_q: float, upper_q: float, device: str):
    totals = defaultdict(lambda: {"missed": 0, "n": 0, "mean_risk": 0, "std_risk": 0})
    per_image_risks = defaultdict(list)
    
    with torch.no_grad():
        im2im_model.eval(); im2im_deep_model.eval(); quantile_model.eval()
        for (noisy, clean) in tqdm(dataloader, desc="Calculating calibration risk"):
            noisy, clean = noisy.to(device), clean.to(device)
            B, C, H, W = noisy.shape
            num_pixels = B * H * W        

            # --------- im2im model ---------
            #assert isinstance(im2im_model.baseModel, Im2Im)
            # pred          = im2im_model(noisy)
            # lower, upper  = return_calibrated_bounds(pred, im2im_lambda)
            # lower, upper  = lower.view(B, 1, H, W), upper.view(B, 1, H, W)             # [B, 1, H, W]
            # missed_bin = get_binary_missed(lower, upper, clean).view(B, -1)          # [B, H * W]
            # per_image_missed_counts = missed_bin.sum(dim=1).cpu()
            # per_image_risks["im2im"].extend((per_image_missed_counts / (H * W)).cpu().numpy())

            # totals["im2im"]["missed"] += per_image_missed_counts.sum().item()
            # totals["im2im"]["n"]      += num_pixels

            # --------- im2im deep model ---------
            assert isinstance(im2im_deep_model.baseModel, UNetModel)
            timevect = torch.full((B,), 0.5, device=device, dtype=torch.float32)
            pred = im2im_deep_model(noisy, timevect)
            lower, upper = return_calibrated_bounds(pred, im2im_deep_lambda)
            lower, upper = lower.view(B, 1, H, W), upper.view(B, 1, H, W)             # [B, 1, H, W]
            missed_bin = get_binary_missed(lower, upper, clean).view(B, -1)          # [B, H * W]
            per_image_missed_counts = missed_bin.sum(dim=1).cpu()
            per_image_risks["im2im_deep"].extend((per_image_missed_counts / (H * W)).cpu().numpy())

            totals["im2im_deep"]["missed"] += per_image_missed_counts.sum().item()
            totals["im2im_deep"]["n"]      += num_pixels

            # --------- quantile model ---------
            quantiles = torch.tensor([lower_q, upper_q], device=device, dtype=torch.float32)
            quantiles = quantiles.unsqueeze(1).expand(-1, B).reshape(-1)                             # [2 * B]
            noisy     = noisy.unsqueeze(0).expand(2, -1, -1, -1, -1).reshape(-1, C, H, W)            # [2 * B, C, H, W]
            pred      = quantile_model(noisy, quantiles).view(2, B, 1, H, W).permute(1, 0, 2, 3, 4)  # [B, 2, 1, H, W]
            lower_pred, upper_pred = pred[:, 0], pred[:, 1]                                          # [B, 1, H, W]
            missed_bin_q = get_binary_missed(lower_pred, upper_pred, clean).view(B, -1)    # [B, H * W]
            per_image_missed_counts_q = missed_bin_q.sum(dim=1)
            per_image_risks["quantile"].extend((per_image_missed_counts_q / (H * W)).cpu().numpy())
            
            totals["quantile"]["missed"] += per_image_missed_counts_q.sum().item()
            totals["quantile"]["n"]      += num_pixels

    for method in totals.keys():
        totals[method]["mean_risk"] = np.mean(per_image_risks[method])
        totals[method]["std_risk"]  = np.std(per_image_risks[method])
    
    del dataloader
    del noisy, clean
    del pred, lower, upper, lower_pred, upper_pred
    del missed_bin, missed_bin_q
    del per_image_missed_counts, per_image_missed_counts_q
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    
    return totals                             
        

def get_test_intervals_missed(test_dataloader: DataLoader, im2im_model: ModelWithUncertainty, im2im_lam: float, 
                              im2im_deep_model: ModelWithUncertainty, im2im_deep_lam: float,
                              quantile_model: UNetModel, lower_q: float, upper_q: float, device: str):
    rows = []
    totals = defaultdict(lambda: {"missed": 0, "n": 0, "mean_risk": 0, "std_risk": 0})
    per_image_risks = defaultdict(list)

    with torch.no_grad():
        im2im_model.eval(); im2im_deep_model.eval(); quantile_model.eval()
        for (noisy, clean) in tqdm(test_dataloader):
            noisy, clean = noisy.to(device), clean.to(device)
            B, C, H, W = noisy.shape
            num_pixels = B * H * W        

            # --------- im2im model ---------
            #assert isinstance(im2im_model.baseModel, Im2Im)
            # pred          = im2im_model(noisy)
            # lower, upper  = return_calibrated_bounds(pred, im2im_lam)
            # lower, upper  = lower.view(B, 1, H, W), upper.view(B, 1, H, W)             # [B, 1, H, W]
            # interval_size = (upper - lower).view(-1)
            # missed_bin    = get_binary_missed(lower, upper, clean).view(B, -1)

            # per_image_missed_counts = missed_bin.sum(dim=1).cpu()
            # per_image_risks["im2im"].extend((per_image_missed_counts / (H * W)).cpu().numpy())

            # totals["im2im"]["missed"] += per_image_missed_counts.sum().item()
            # totals["im2im"]["n"]      += num_pixels

            # rows.append(
            #     pd.DataFrame({
            #         "method":   "im2im",
            #         "interval": interval_size.cpu().numpy().astype('float32'),
            #         "missed":   missed_bin.view(-1).cpu().numpy().astype('bool'),}))
            
            # --------- im2im deep model ---------
            assert isinstance(im2im_deep_model.baseModel, UNetModel)
            timevect = torch.full((B,), 0.5, device=device, dtype=torch.float32)
            pred = im2im_deep_model(noisy, timevect)
            lower, upper = return_calibrated_bounds(pred, im2im_deep_lam)
            lower, upper = lower.view(B, 1, H, W), upper.view(B, 1, H, W)             # [B, 1, H, W]
            interval_size = (upper - lower).view(-1)
            missed_bin = get_binary_missed(lower, upper, clean).view(B, -1)          # [B, H * W]
            
            per_image_missed_counts = missed_bin.sum(dim=1).cpu()
            per_image_risks["im2im_deep"].extend((per_image_missed_counts / (H * W)).cpu().numpy())
            
            totals["im2im_deep"]["missed"] += per_image_missed_counts.sum().item()
            totals["im2im_deep"]["n"]      += num_pixels

            rows.append(
                pd.DataFrame({
                    "method":   "im2im_deep",
                    "interval": interval_size.cpu().numpy().astype('float32'),
                    "missed":   missed_bin.view(-1).cpu().numpy().astype('bool'),}))
            
            # --------- quantile model ---------
            quantiles = torch.tensor([lower_q, upper_q], device=device, dtype=torch.float32)
            quantiles = quantiles.unsqueeze(1).expand(-1, B).reshape(-1)                             # [2 * B]
            noisy     = noisy.unsqueeze(0).expand(2, -1, -1, -1, -1).reshape(-1, C, H, W)            # [2 * B, C, H, W] 
            pred      = quantile_model(noisy, quantiles).view(2, B, 1, H, W).permute(1, 0, 2, 3, 4)  # [B, 2, 1, H, W]
            
            lower_pred, upper_pred = pred[:, 0], pred[:, 1]                                          # [B, 1, H, W]
            interval_size_q        = (upper_pred - lower_pred).view(-1)
            missed_bin_q           = get_binary_missed(lower_pred, upper_pred, clean).view(B, -1)    # [B, H * W]

            per_image_missed_counts_q = missed_bin_q.sum(dim=1)
            per_image_risks["quantile"].extend((per_image_missed_counts_q / (H * W)).cpu().numpy())

            totals["quantile"]["missed"] += per_image_missed_counts_q.sum().item()
            totals["quantile"]["n"]      += num_pixels

            rows.append(
                pd.DataFrame({
                    "method":   "quantile",
                    "interval": interval_size_q.cpu().numpy().astype('float32'),
                    "missed":   missed_bin_q.view(-1).cpu().numpy().astype('bool'),}))

    for method in totals.keys():
        totals[method]["mean_risk"] = np.mean(per_image_risks[method])
        totals[method]["std_risk"]  = np.std(per_image_risks[method])

    df_intervals_missed = pd.concat(rows, ignore_index=True, copy=False)
    df_intervals_missed["method"] = df_intervals_missed["method"].astype("category")

    del test_dataloader                       
    del noisy, clean                          
    del pred, lower, upper, lower_pred, upper_pred
    del interval_size, interval_size_q, missed_bin, missed_bin_q
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    return df_intervals_missed, totals

def get_largest_residual_indices(model: UNetModel, dataloader: DataLoader, 
                                 device: torch.device, top_k: 10, 
                                 metric: Literal["l2", "l1", "ssim"]="l2") -> List[int]:
    """
    Evaluate `model` on `dataloader` once and return the dataset indices of the `top_k`
    samples with the largest residual (per–image error).
    """
    model.eval()
    errs = []
    if metric == "l2":
        loss = nn.MSELoss(reduction="none")
    elif metric == "l1":
        loss = nn.L1Loss(reduction="none")
    elif metric == "ssim":
        loss = StructuralSimilarityIndexMeasure(data_range=1.0, reduction="none")
    elif metric == "lpips":
        loss = LPIPS(net="vgg")
    else:
        raise ValueError(f"Unknown metric: {metric}")
    loss = loss.to(device)

    with torch.no_grad():
        for noisy, clean in tqdm(dataloader):
            noisy, clean = noisy.to(device), clean.to(device)
            B = noisy.shape[0]

            quantiles = torch.full((B,), 0.5, device=device, dtype=torch.float32)
            pred      = model(noisy, quantiles)

            error = loss(pred, clean).view(B, -1).mean(dim=1)
            errs.append(error.cpu())
    
    errs = torch.cat(errs, dim=0)
    if metric == "ssim":
        worst_vals, worst_pos = torch.topk(errs, top_k, dim=0, largest=False, sorted=True)
    else:
        worst_vals, worst_pos = torch.topk(errs, top_k, dim=0, largest=True, sorted=True)

    del noisy, clean, quantiles, pred, error
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()

    return worst_pos, worst_vals.tolist()
            
            