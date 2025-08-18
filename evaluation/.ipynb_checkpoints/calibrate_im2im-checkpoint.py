import copy
import inspect
import os
import pdb
import sys
import gc

import numpy as np
import psutil
import torch
from scipy.optimize import brentq
from scipy.stats import binom, spearmanr
from torch.utils.data import DataLoader, Dataset, TensorDataset
from tqdm import tqdm
from models.pinball import PinballLoss
import torch.nn as nn

from models.im2im.add_uncertainty_im2im import ModelWithUncertainty
from models.quantile_uqnet import UNetModel
from evaluation import return_calibrated_bounds

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
            print(f"\nBRENTQ RUNTIME ERROR at muhat={muhat:.6f}")
            return 1.0

# Always outputs [0,1] valued nested sets
def nested_sets_from_output(model: ModelWithUncertainty, output, lam=None):
    lower_edge, prediction, upper_edge = quantile_regression_nested_sets_from_output(model, output, lam)
    upper_edge = torch.maximum(upper_edge, prediction + 1e-6) # set a lower bound on the size.
    lower_edge = torch.minimum(lower_edge, prediction - 1e-6)

    return lower_edge, prediction, upper_edge 

def quantile_regression_nested_sets_from_output(model: ModelWithUncertainty, output, lam=None):
    if lam == None:
        if model.lhat == None:
            raise Exception("You have to specify lambda unless your model is already calibrated.")
        lam = model.lhat
    output[:,0,:,:] = torch.minimum(output[:,0,:,:], output[:,1,:,:] - 1e-6)
    output[:,2,:,:] = torch.maximum(output[:,2,:,:], output[:,1,:,:] + 1e-6)
    upper_edge = lam * (output[:,2,:,:] - output[:,1,:,:]) + output[:,1,:,:] 
    lower_edge = output[:,1,:,:] - lam * (output[:,1,:,:] - output[:,0,:,:])

    return lower_edge, output[:,1,:,:], upper_edge 
  
def fraction_missed_loss(pset,label):
    misses = (pset[0].squeeze() > label.squeeze()).float() + (pset[2].squeeze() < label.squeeze()).float()
    misses[misses > 1.0] = 1.0
    d = len(misses.shape)
    return misses.mean(dim=tuple(range(1,d)))

def calibrate_model(model: ModelWithUncertainty, dataset: Dataset, config):
    with torch.no_grad():
        model.eval()
        alpha = config['alpha']
        delta = config['delta']
        device = config['device']
        print("Initialize lambdas")
        lambdas = torch.linspace(config['minimum_lambda'],config['maximum_lambda'],config['num_lambdas'])
        print("Initialize labels")
        labels_shape = list(dataset[0][1].unsqueeze(0).shape)
        labels_shape[0] = len(dataset)
        labels = torch.zeros(tuple(labels_shape), device='cpu')
        outputs_shape = list(model(dataset[0][0].unsqueeze(0).to(device)).shape)
        outputs_shape[0] = len(dataset)
        outputs = torch.zeros(tuple(outputs_shape),device='cpu')
        print("Collecting dataset")
        tempDL = DataLoader(dataset, num_workers=0, batch_size=config['batch_size'], pin_memory=True) 
        counter = 0
        for batch in tqdm(tempDL):
            outputs[counter:counter+batch[0].shape[0],:,:,:,:] = model(batch[0].to(device)).cpu()
            labels[counter:counter+batch[1].shape[0]] = batch[1]
            counter += batch[0].shape[0]
        print("Output dataset")
        out_dataset = TensorDataset(outputs, labels.cpu())
        dlambda = lambdas[1] - lambdas[0]
        model.set_lhat(lambdas[-1] + dlambda - 1e-9)
        print("Computing losses")
        calib_loss_table = torch.zeros((outputs.shape[0], lambdas.shape[0]))
        for lam in reversed(lambdas):
            losses = get_rcps_losses_from_outputs(model, out_dataset, lam - dlambda, device)
            calib_loss_table[:,np.where(lambdas==lam)[0]] = losses[:,None]
            Rhat = losses.mean()
            RhatPlus = HB_mu_plus(Rhat.item(), losses.shape[0], delta)
            print(f"\rLambda: {lam:.4f}  |  Rhat: {Rhat:.4f}  |  RhatPlus: {RhatPlus:.4f}  ",end='')
            if Rhat >= alpha or RhatPlus > alpha:
                model.set_lhat(lam)
                print("")
                print(f"Model's lhat set to {model.lhat}")
                break
        return model, calib_loss_table
    
def get_rcps_losses_from_outputs(model, out_dataset, lam, device):
    losses = []
    dataloader = DataLoader(out_dataset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True) 
    model = model.to(device)
    for batch in dataloader:
        x, labels = batch
        sets = nested_sets_from_output(model, x.to(device), lam) 
        losses = losses + [fraction_missed_loss(sets, labels.to(device)).cpu(),]
    return torch.cat(losses,dim=0)


def get_binary_missed(lower_bound, upper_bound, gt):
    missed = (gt < lower_bound) | (gt > upper_bound)
    return missed.to(lower_bound.dtype)

def compute_optimal_lambdas(dataloader: DataLoader, model: ModelWithUncertainty, 
                            alpha=0.1, min_lam=0, max_lam=6, num_lam=1000, 
                            device='cuda'):
    alpha = alpha - (1 - alpha) / len(dataloader.dataset)
    delta = 0.1
    lambdas = torch.linspace(min_lam, max_lam, num_lam).to(device)
    dlambda = lambdas[1] - lambdas[0]

    all_outputs = []
    all_labels = []
    print("Collecting model outputs and labels...")
    with torch.no_grad():
        model.eval()
        for (noisy, clean) in tqdm(dataloader):
            noisy, clean = noisy.to(device), clean.to(device)
            if isinstance(model.baseModel, UNetModel):
                timevect = torch.full((noisy.shape[0],), 0.5, device=device, dtype=torch.float32)
                out = model(noisy, timevect)
            else:
                out = model(noisy)
            
            all_outputs.append(out)
            all_labels.append(clean)
    
    all_outputs = torch.cat(all_outputs, dim=0)
    print("All outputs shape", all_outputs.shape)
    all_labels = torch.cat(all_labels, dim=0)
    n_samples = all_labels.size(0)

    optimal_lambda = max_lam + dlambda - 1e-9
    print("Evaluating lambdas...")
    for lambda_ in reversed(lambdas):
        sets = nested_sets_from_output(model, all_outputs, lambda_)
        losses = fraction_missed_loss(sets, all_labels)
        # lower, upper = return_calibrated_bounds(all_outputs, lambda_)
        # B, H, W = sets[0].shape
        # lower, upper = lower.view(B, 1, H, W), upper.view(B, 1, H, W)
        # print(lower.shape, upper.shape)
        # missed_bin = get_binary_missed(lower, upper, all_labels).view(B, -1)
        # per_image_missed_counts = missed_bin.sum(dim=1).sum().item()
        # losses_ours = per_image_missed_counts / (B * H * W)
        Rhat = losses.mean()
        RhatPlus = HB_mu_plus(Rhat.item(), n_samples, delta)
        print(f"Lambda: {lambda_:.4f}  |  Rhat: {Rhat:.4f}  |  RhatPlus: {RhatPlus:.4f}")
        # if Rhat > alpha:
        if Rhat >= alpha:
            optimal_lambda = lambda_
            print(f"Optimal lambda found: {optimal_lambda:.4f}")
            break
    
    del out, all_labels, all_outputs, noisy, clean
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    
    return optimal_lambda.item()
    

