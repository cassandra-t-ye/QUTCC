import gc
from contextlib import suppress
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import brentq
from scipy.special import xlogy
from scipy.stats import binom
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.im2im.add_uncertainty_im2im import ModelWithUncertainty
from models.quantile_uqnet import UNetModel

def h1(y, mu):
    # xlogy applies the 0 * log(0) = 0 convention, so y = 0 (or y = 1) stays finite instead
    # of turning the whole tail bound into a NaN.
    return xlogy(y, y / mu) + xlogy(1 - y, (1 - y) / (1 - mu))

### Log tail inequalities of mean
def hoeffding_plus(mu, x, n):
    return -n * h1(np.minimum(mu, x), mu)

def bentkus_plus(mu, x, n):
    return np.log(max(binom.cdf(np.floor(n * x), n, mu), 1e-10)) + 1

def HB_mu_plus(muhat, n, delta, maxiters=1000):
    muhat = max(muhat, 1e-12)  # keep the brentq bracket strictly inside (0, 1)
    def _tailprob(mu):
        hoeffding_mu = hoeffding_plus(mu, muhat, n)
        bentkus_mu = bentkus_plus(mu, muhat, n)
        return min(hoeffding_mu, bentkus_mu) - np.log(delta)
    if _tailprob(1 - 1e-10) > 0:
        return 1.0
    try:
        return brentq(_tailprob, muhat, 1 - 1e-10, maxiter=maxiters)
    except (RuntimeError, ValueError):
        print(f"\nBRENTQ RUNTIME ERROR at muhat={muhat:.6f}")
        return 1.0

# Always outputs [0,1] valued nested sets
def nested_sets_from_output(model: ModelWithUncertainty, output, lam=None):
    lower_edge, prediction, upper_edge = quantile_regression_nested_sets_from_output(model, output, lam)
    upper_edge = torch.maximum(upper_edge, prediction + 1e-6) # set a lower bound on the size.
    lower_edge = torch.minimum(lower_edge, prediction - 1e-6)
    return lower_edge, prediction, upper_edge 

def quantile_regression_nested_sets_from_output(model: ModelWithUncertainty, output, lam=None):
    if lam is None:
        if model.lhat is None:
            raise ValueError("Lambda must be specified if the model is not calibrated.")
        lam = model.lhat
    output[:, 0, :, :] = torch.minimum(output[:, 0, :, :], output[:, 1, :, :] - 1e-6)
    output[:, 2, :, :] = torch.maximum(output[:, 2, :, :], output[:, 1, :, :] + 1e-6)
    upper_edge = lam * (output[:, 2, :, :] - output[:, 1, :, :]) + output[:, 1, :, :]
    lower_edge = output[:, 1, :, :] - lam * (output[:, 1, :, :] - output[:, 0, :, :])
    return lower_edge, output[:, 1, :, :], upper_edge
  
def fraction_missed_loss(pset,label):
    misses = (pset[0].squeeze() > label.squeeze()).float() + (pset[2].squeeze() < label.squeeze()).float()
    misses[misses > 1.0] = 1.0
    return misses.mean(dim=tuple(range(1, misses.ndim)))

def compute_optimal_lambdas(dataloader: DataLoader, model: ModelWithUncertainty,
                            alpha: float = 0.1, min_lam: float = 0, max_lam: float = 6, num_lam: int = 1000,
                            device: str = 'cuda') -> float:
    alpha = alpha - (1 - alpha) / len(dataloader.dataset)
    delta = 0.1
    lambdas = torch.linspace(min_lam, max_lam, num_lam, device=device)
    dlambda = lambdas[1] - lambdas[0]

    all_outputs, all_labels = [], []
    with torch.no_grad():
        model.eval()
        for noisy, clean in tqdm(dataloader, desc="Collecting model outputs"):
            noisy, clean = noisy.to(device), clean.to(device)
            if isinstance(model.baseModel, UNetModel):
                timevect = torch.full((noisy.shape[0],), 0.5, device=device, dtype=torch.float32)
                out = model(noisy, timevect)
            else:
                out = model(noisy)
            
            all_outputs.append(out.cpu())
            all_labels.append(clean.cpu())
    
    all_outputs = torch.cat(all_outputs, dim=0).to(device)
    all_labels = torch.cat(all_labels, dim=0).to(device)
    n_samples = all_labels.size(0)

    optimal_lambda = max_lam + dlambda - 1e-9
    print("Evaluating lambdas...")
    last_lambda = None
    for lambda_ in reversed(lambdas):
        sets = nested_sets_from_output(model, all_outputs, lambda_)
        losses = fraction_missed_loss(sets, all_labels)
        Rhat = losses.mean().item()
        RhatPlus = HB_mu_plus(Rhat, n_samples, delta)
        print(f"Lambda: {lambda_:.4f}  |  Rhat: {Rhat:.4f}  |  RhatPlus: {RhatPlus:.4f}")
        if Rhat >= alpha:
            optimal_lambda = last_lambda if last_lambda is not None else lambda_
            print(f"Optimal lambda found: {optimal_lambda:.4f}")
            break
        last_lambda = lambda_
    
    del all_outputs, all_labels
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    
    return optimal_lambda.item()

def quantile_triple_from_model(model: UNetModel, noisy: torch.Tensor,
                               q_lo: float = 0.05, q_mid: float = 0.5, q_hi: float = 0.95,
                               device: str = 'cuda') -> tuple:
    """
    Query the quantile network at three quantile levels and return the heuristic
    interval (lower, prediction, upper), each [B, 1, H, W].

    Quantile crossings are repaired the same way `nested_sets_from_output` does, so the
    interval is always strictly nested around the median prediction.
    """
    B, C, H, W = noisy.shape
    qs = torch.tensor([q_lo, q_mid, q_hi], device=device, dtype=torch.float32)
    qs = qs.unsqueeze(1).expand(-1, B).reshape(-1)                                   # [3 * B]
    inp = noisy.unsqueeze(0).expand(3, -1, -1, -1, -1).reshape(-1, C, H, W)          # [3 * B, C, H, W]
    pred = model(inp, qs).view(3, B, -1, H, W)                                       # [3, B, 1, H, W]

    lower, mid, upper = pred[0], pred[1], pred[2]
    lower = torch.minimum(lower, mid - 1e-6)
    upper = torch.maximum(upper, mid + 1e-6)
    return lower, mid, upper

def required_lambda(lower: torch.Tensor, pred: torch.Tensor,
                    upper: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Smallest lambda >= 0 for which the RCPS-scaled interval
        [pred - lam * (pred - lower),  pred + lam * (upper - pred)]
    covers `target`. A pixel is missed at lambda iff required_lambda > lambda, so one
    pass over the data gives the exact risk curve for *every* lambda at once.
    """
    below = (pred - target) / (pred - lower)
    above = (target - pred) / (upper - pred)
    return torch.clamp(torch.maximum(below, above), min=0.0)

def compute_rcps_lambda_quantile(dataloader: DataLoader, model: UNetModel,
                                 alpha: float = 0.1, delta: float = 0.1,
                                 q_lo: float = 0.05, q_mid: float = 0.5, q_hi: float = 0.95,
                                 min_lam: float = 0.0, max_lam: float = 6.0, num_lam: int = 1000,
                                 use_ucb: bool = True, device: str = 'cuda') -> Dict[str, Any]:
    """
    Calibrate the QUTCC quantile network with RCPS instead of QUTCC's own quantile search.

    The heuristic interval is the network's `q_lo`/`q_hi` prediction (5th/95th percentile by
    default) around its `q_mid` prediction; RCPS then picks the smallest scalar lambda whose
    risk stays below alpha:
        lambda_hat = inf { lam : Rhat+(lam') <= alpha for all lam' >= lam }
    with Rhat the mean per-image miss fraction and Rhat+ its Hoeffding-Bentkus UCB at level
    delta. Set `use_ucb=False` to select on the plain empirical risk instead (this is what
    `compute_optimal_lambdas` does for the im2im nets).

    Returns a dict with `lambda`, the full risk curve, and the risk achieved at lambda_hat.
    """
    alpha_adj = alpha - (1 - alpha) / len(dataloader.dataset)
    lambdas = torch.linspace(min_lam, max_lam, num_lam, dtype=torch.float64)
    K = num_lam
    print(f"RCPS calibration on q=({q_lo}, {q_mid}, {q_hi}) | alpha={alpha:.5g} "
          f"(adjusted {alpha_adj:.5g}) | delta={delta:.5g}")

    lambdas_dev = lambdas.to(device=device, dtype=torch.float32)
    per_image_counts = []
    with torch.no_grad():
        model.eval()
        for noisy, clean in tqdm(dataloader, desc="Collecting required lambdas"):
            noisy, clean = noisy.to(device), clean.to(device)
            B = noisy.shape[0]

            lower, mid, upper = quantile_triple_from_model(model, noisy, q_lo, q_mid, q_hi, device)
            lam_req = required_lambda(lower, mid, upper, clean).reshape(B, -1)        # [B, H * W]

            # bucket index b = #{grid points strictly below lam_req}, so the pixel is missed
            # at lambdas[k] iff b >= k + 1.
            b = torch.bucketize(lam_req, lambdas_dev, right=False)                    # [B, H * W] in [0, K]
            counts = torch.zeros(B, K + 1, device=device, dtype=torch.float32)
            counts.scatter_add_(1, b, torch.ones_like(b, dtype=torch.float32))
            per_image_counts.append(counts.cpu())

    counts = torch.cat(per_image_counts, dim=0).double()                              # [N, K + 1]
    n_samples = counts.shape[0]
    n_pixels = counts.sum(dim=1, keepdim=True)

    # suffix[:, k] = #{pixels with b >= k}; risk at lambdas[k] uses suffix[:, k + 1]
    suffix = counts.flip(1).cumsum(dim=1).flip(1)
    per_image_risk = suffix[:, 1:] / n_pixels                                         # [N, K]
    Rhat = per_image_risk.mean(dim=0).numpy()

    print(f"Evaluating {K} lambdas over {n_samples} calibration images...")
    RhatPlus = np.full(K, np.nan)
    lambda_hat, idx_hat = None, None
    for k in range(K - 1, -1, -1):
        RhatPlus[k] = HB_mu_plus(float(Rhat[k]), n_samples, delta)
        crit = RhatPlus[k] if use_ucb else Rhat[k]
        if k % 50 == 0 or crit >= alpha_adj:
            print(f"Lambda: {lambdas[k]:.4f}  |  Rhat: {Rhat[k]:.4f}  |  RhatPlus: {RhatPlus[k]:.4f}")
        if crit >= alpha_adj:
            if k + 1 < K:
                idx_hat = k + 1
                lambda_hat = float(lambdas[k + 1])
            break
        idx_hat, lambda_hat = k, float(lambdas[k])

    if lambda_hat is None:
        lambda_hat = float(max_lam + (lambdas[1] - lambdas[0]))
        idx_hat = None
        print(f"Warning: no lambda in [{min_lam}, {max_lam}] satisfied the risk constraint; "
              f"returning {lambda_hat:.4f} (uncalibrated).")
    else:
        print(f"Optimal lambda found: {lambda_hat:.4f}")

    result = {
        "lambda": lambda_hat,
        "alpha": alpha, "alpha_adjusted": float(alpha_adj), "delta": delta,
        "q_lo": q_lo, "q_mid": q_mid, "q_hi": q_hi,
        "use_ucb": use_ucb, "n_calib_images": int(n_samples),
        "calib_risk": float(Rhat[idx_hat]) if idx_hat is not None else None,
        "calib_risk_ucb": float(RhatPlus[idx_hat]) if idx_hat is not None else None,
        "calib_risk_std": float(per_image_risk[:, idx_hat].std().item()) if idx_hat is not None else None,
        "lambdas": lambdas.numpy().tolist(),
        "risk_curve": Rhat.tolist(),
        "risk_ucb_curve": RhatPlus.tolist(),
    }

    del counts, suffix, per_image_risk, per_image_counts
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

    return result

def compute_optimal_lambdas_quantile(dataloader: DataLoader, model: UNetModel, alpha: float,
                                     lower_min: float, lower_max: float, upper_min: float, upper_max: float,
                                     max_iterations: int = 20, device: str = 'cuda') -> Dict[str, float]:
    """Compute optimal lambda parameters for quantile regression using a dataset."""
    print(f"Computing optimal quantile bounds for alpha={alpha:.5g}...")
    # Calculate adjusted alpha with correction for finite sample size
    alpham = (alpha - (1 - alpha) / len(dataloader.dataset)) / 2
    print(f"Target miss rate (per side): {alpham:.5g}")

    l_min, l_max = lower_min, lower_max
    u_min, u_max = upper_min, upper_max
    last_valid_lower, last_valid_upper = None, None
    tol = 1e-3

    with torch.no_grad():
        model.eval()
        for i in range(max_iterations):
            lower_q = (l_min + l_max) / 2
            upper_q = (u_min + u_max) / 2
            missed_low_cnt, missed_high_cnt, total_pixels = 0., 0., 0.
            
            for noisy, clean in tqdm(dataloader, desc=f"Iter {i+1}/{max_iterations}"):
                noisy, clean = noisy.to(device), clean.to(device)
                B, C, H, W = noisy.shape
                total_pixels += B * H * W

                quantiles = torch.tensor([lower_q, upper_q], device=device, dtype=torch.float32)
                quantiles = quantiles.unsqueeze(1).expand(-1, B).reshape(-1)                            # [2 * B]
                noisy = noisy.unsqueeze(0).expand(2, -1, -1, -1, -1).reshape(-1, C, H, W)               # [2 * B, C, H, W]
                pred = model(noisy, quantiles).view(2, B, 1, H, W).permute(1, 0, 2, 3, 4)               # [B, 2, 1, H, W]

                missed_low_cnt += (pred[:, 0] > clean).float().sum().item()                             # [B, 1, H, W]
                missed_high_cnt += (pred[:, 1] < clean).float().sum().item()

            missed_low = missed_low_cnt / total_pixels
            missed_high = missed_high_cnt / total_pixels
            low_status = "✅" if missed_low <= alpham else "❌"
            high_status = "✅" if missed_high <= alpham else "❌"
            
            # Formats to 5 significant figures and uses scientific notation for small numbers.
            print(f"Iter {i+1:2d}: lower_q={lower_q:<12.5g} (risk: {missed_low:<12.5g} {low_status}) | "
                  f"upper_q={upper_q:<12.5g} (risk: {missed_high:<12.5g} {high_status})")
            
            if missed_low > alpham: l_max = lower_q                 # lower too high -> move down
            else: l_min = lower_q; last_valid_lower = lower_q       # satisfies alpha -> move up

            if missed_high > alpham: u_min = upper_q                # upper too low -> move up
            else: u_max = upper_q; last_valid_upper = upper_q       # satisfies alpha -> move down

            if (l_max - l_min) < tol and (u_max - u_min) < tol and last_valid_lower and last_valid_upper:
                print("Tolerance reached – stopping search.")
                break            
        
    gc.collect()
    torch.cuda.empty_cache()  
    torch.cuda.ipc_collect()  
    
    if last_valid_lower is None or last_valid_upper is None:
        print("Warning: No valid bounds found within maximum iterations")
        return {"lower_q": lower_q, "upper_q": upper_q}
    print(f"Final valid bounds: lower={last_valid_lower}, upper={last_valid_upper}")
    return {"lower_q": last_valid_lower, "upper_q": last_valid_upper}