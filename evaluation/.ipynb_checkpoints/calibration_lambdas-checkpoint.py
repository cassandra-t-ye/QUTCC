import gc
from contextlib import suppress

import numpy as np
import torch
from scipy.optimize import brentq
from scipy.stats import binom
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from models.im2im.add_uncertainty_im2im import ModelWithUncertainty
from models.quantile_uqnet import UNetModel


def h1(y, mu):
    return y*np.log(y/mu) + (1-y)*np.log((1-y)/(1-mu))
    
### Log tail inequalities of mean
def hoeffding_plus(mu, x, n):
    return -n * h1(np.minimum(mu,x),mu)

def bentkus_plus(mu, x, n):
    return np.log(max(binom.cdf(np.floor(n*x),n,mu),1e-10))+1

def HB_mu_plus(muhat, n, delta, maxiters=1000):
    def _tailprob(mu):
        hoeffding_mu = hoeffding_plus(mu, muhat, n)
        bentkus_mu = bentkus_plus(mu, muhat, n)
        return min(hoeffding_mu, bentkus_mu) - np.log(delta)
    if _tailprob(1-1e-10) > 0:
        return 1
    else:
        try:
            return brentq(_tailprob, muhat, 1-1e-10, maxiter=maxiters)
        except:
            # TODO: figure out why this is happening
            print(f"\nBRENTQ RUNTIME ERROR at muhat={muhat:.6f}")
            return 1.0

def fraction_missed_loss(lower, upper,label):
    misses = (lower > label.squeeze()).float() + (upper < label.squeeze()).float()
    misses[misses > 1.0] = 1.0
    d = len(misses.shape)
    return misses.mean(dim=tuple(range(1,d)))

def fraction_missed_loss_im2im(pset,label):
  misses = (pset[0].squeeze() > label.squeeze()).float() + (pset[2].squeeze() < label.squeeze()).float()
  misses[misses > 1.0] = 1.0
  d = len(misses.shape)
  return misses.mean(dim=tuple(range(1,d)))

    
def compute_optimal_lambdas(dataloader, model: ModelWithUncertainty, alpha, min_lam, max_lam, num_lam, device='cuda'):
    """
    Computes the optimal lambda for the given dataset.
    
    Parameters:
    - dataset (torch.utils.data.Dataset): Dataset containing pairs of clean and noisy images.
    - model (callable): The model used to denoise the images.
    - device (str): The device to run the model on (e.g. 'cuda' or 'cpu').
    - alpha (float): Target risk level for calibration.
    
    Returns:
    - float: The optimal lambda value.
    """
    
    model.eval()
    outputs = []
    labels = []

    with torch.no_grad():
        for noisy, clean in tqdm(dataloader, desc="Collecting model outputs"):
            batch_size = noisy.shape[0]
            noisy, clean = noisy.to(device), clean.to(device)
            if isinstance(model.baseModel, UNetModel):
                timevect = torch.full((batch_size,), 0.5, device=device, dtype=torch.float32)
                out = model(noisy, timevect)  # out: [B, 3, H, W]
            else:
                out = model(noisy)  # out: [B, 3, H, W]
            #print("out shape = ", out.shape)

            lower = out[:, 0, :, :]
            mean  = out[:, 1, :, :]
            upper = out[:, 2, :, :]

            stacked = torch.stack([lower, mean, upper])  # shape: (3, B, H, W)
            outputs.append(stacked)
            labels.append(clean)

        # shape: (3, N, H, W)
        outputs = torch.cat(outputs, dim=1)
        labels = torch.cat(labels, dim=0)  # shape: (N, H, Wx)

        # Create lambda grid
        lambdas = torch.linspace(min_lam, max_lam, num_lam)
        loss_table = torch.zeros((labels.shape[0], num_lam))  # (N, num_lam)

        #print("outputs = ", outputs.shape)
        #print("labels = ", labels.shape)
        # Evaluate loss for each lambda
        for i, lam in enumerate(tqdm(lambdas, desc="Evaluating lambdas")):
            curr_sets = torch.stack([
                outputs[1] - lam * (outputs[1] - outputs[0]),  # lower
                outputs[1],                                   # center
                outputs[1] + lam * (outputs[2] - outputs[1])  # upper
            ])  # shape: (3, N, H, W)

            loss = fraction_missed_loss_im2im(curr_sets, labels)  # shape: (N,)
            loss_table[:, i] = loss

        # Hoeffding-Bentkus bound
        n = len(dataloader.dataset)
        Rhats = loss_table.mean(dim=0)
        RhatPlus = torch.tensor([HB_mu_plus(Rhat.item(), n, 0.1) for Rhat in Rhats])
        valid = RhatPlus <= alpha

        if valid.any():
            idx = torch.argmax(valid.int())
            optimal_lambda = lambdas[idx]
            print(f"Calibrated lambda = {optimal_lambda.item()}")
        else:
            print("No lambda satisfied the calibration bound. Returning most conservative value.")
            optimal_lambda = lambdas[-1]

    del outputs, labels, loss_table, noisy, clean, out, lower, mean, upper
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()
    return optimal_lambda.item()



def fraction_missed_loss_quantile(lower, upper, label, q):
    """Calculate the fraction of missed predictions based on quantile bounds.
    
    Args:
        lower: Lower bound predictions
        upper: Upper bound predictions
        label: Ground truth labels
        q (str): Either "low" or "high" to specify which bound to check
    
    Returns:
        Tensor: Mean violation rate across spatial dimensions
    """
    lower_violation = (lower.squeeze() > label.squeeze()).float()
    upper_violation = (upper.squeeze() < label.squeeze()).float()
    
    if q == "low":
        return lower_violation.mean(dim=tuple(range(1, len(lower_violation.shape))))
    if q == "high":
        return upper_violation.mean(dim=tuple(range(1, len(upper_violation.shape))))


def compute_optimal_lambdas_quantile(dataloader: DataLoader, model: UNetModel, alpha: float, 
                                     lower_min: float, lower_max: float, upper_min: float, upper_max: float, 
                                     max_iterations: int = 20, device: str = 'cuda'):
    """Compute optimal lambda parameters for quantile regression using a dataset.
    
    Args:
        dataset (torch.utils.data.Dataset): Dataset containing pairs of noisy and clean images
        quantile_model: Model that produces quantile predictions
        alpha (float): Target error rate
        max_iterations (int): Maximum number of binary search iterations
        device (str): Device to run computations on
        
    Returns:
        tuple: (lower_quantile, upper_quantile) optimal values
    """
    # Calculate adjusted alpha with correction for finite sample size
    alpham = (alpha - (1 - alpha) / len(dataloader.dataset)) / 2
    print(f"Target alpha: {alpha}, adjusted alpha_m: {alpham}")

    lower = (lower_min + lower_max) / 2
    upper = (upper_min + upper_max) / 2
    last_valid_lower, last_valid_upper = None, None
    iteration = 0
    tol = 1e-5

    with torch.no_grad():
        model.eval()        
        while iteration < max_iterations:
            print(f"Iteration {iteration + 1}: lower={lower:.6f}, upper={upper:.6f}")
            missed_low_cnt, missed_high_cnt, total_pixels = 0., 0., 0.
            
            for (noisy, clean) in tqdm(dataloader, desc=f"Iteration {iteration+1}/{max_iterations}"):
                noisy, clean = noisy.to(device), clean.to(device)
                B, C, H, W = noisy.shape
                total_pixels += B * H * W
                
                quantiles = torch.tensor([lower, upper], device=device, dtype=torch.float32)
                quantiles = quantiles.unsqueeze(1).expand(-1, B).reshape(-1)                            # [2 * B]
                noisy = noisy.unsqueeze(0).expand(2, -1, -1, -1, -1).reshape(-1, C, H, W)               # [2 * B, C, H, W]
                pred = model(noisy, quantiles)                                                          # [2 * B, C, H, W]
                pred = pred.view(2, B, 1, H, W).permute(1, 0, 2, 3, 4)                                  # [B, 2, 1, H, W]

                lower_pred, upper_pred = pred[:, 0], pred[:, 1]                                         # [B, 1, H, W]
                missed_low_cnt += (lower_pred > clean).float().sum().item()
                missed_high_cnt += (upper_pred < clean).float().sum().item()

            missed_low = missed_low_cnt / total_pixels
            missed_high = missed_high_cnt / total_pixels
            
            if missed_low > alpham: lower_max = lower               # lower too high -> move down
            else: last_valid_lower = lower; lower_min = lower       # satisfies alpha -> move up
            lower = 0.5 * (lower_min + lower_max)

            if missed_high > alpham: upper_min = upper              # upper too low -> move up
            else: last_valid_upper = upper; upper_max = upper       # satisfies alpha -> move down
            upper = 0.5 * (upper_min + upper_max)
            
            print(f"Iteration {iteration + 1}: low_risk: {missed_low:.6f}, high_risk: {missed_high:.6f}")
            if ((upper_max - upper_min) < tol and (lower_max - lower_min) < tol):
                print("Tolerance reached – stopping search.")
                break            
            iteration += 1
        
    with suppress(UnboundLocalError):
        del noisy, clean, quantiles, pred, lower_pred, upper_pred
    torch.cuda.empty_cache()  
    torch.cuda.ipc_collect()   
    gc.collect()  
    
    if last_valid_lower is None or last_valid_upper is None:
        print("Warning: No valid bounds found within maximum iterations")
        return {"lower_q": lower, "upper_q": upper}
    else:
        print(f"Final valid bounds: lower={last_valid_lower}, upper={last_valid_upper}")
        return {"lower_q": last_valid_lower, "upper_q": last_valid_upper}
        
    
