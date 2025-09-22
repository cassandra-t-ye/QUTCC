import json
import random
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib.colors import to_hex, to_rgb
from matplotlib.patches import Patch
from torch import nn
from tqdm import tqdm

from models.checkpointing import load_checkpoint_for_inference
from evaluation import (check_quantile_crossings, get_risk,
                        size_stratified_risk)
from evaluation.calibration import (compute_optimal_lambdas,
                                    compute_optimal_lambdas_quantile)
from evaluation.calibration_utils import (CalibrationManager,
                                          get_calib_dataloader,
                                          AnalysisManager)
from utils import _prepare_for_json, json_converter, print_memory_stats

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

ROOT = "/share/monakhova/Cassandra_data/UQNet_proj"
# ROOT = "/scratch2/UQNet_proj"

BEST_RUNS = {"gaussian": {"experiment_type": "gaussian",
                          "im2im_root": "im2im_epochs50_bs16_2025-07-05-22.51.54",
                          "im2im_epoch": 35,
                        #   "quantile_root": "unet_quantile_epochs100_bs4_2025-06-26-01.11.24",
                          "unet_im2im_root": "unet_im2im_epochs100_bs4_2025-06-26-04.33.21",
                          "unet_im2im_epoch": 15,
                          "quantile_root": "unet_quantile_epochs50_bs4_2025-05-19-10.11.52",
                          "quantile_epoch": 20,
                          "min_noise": 0,
                          "max_noise": 0.5,
                          "num_noise": 40,
                          "calib_data_path": ROOT + "/Denoising_FMD/calibration",
                          "test_data_path": ROOT + "/Denoising_FMD/validation",
                          "in_channels": 1,
                          "batch_size": 20,
                          "experiments_folder": ROOT + "/experiments"},
            "poisson": {"experiment_type": "poisson",
                        "im2im_root": "im2im_epochs50_bs16_2025-07-05-22.52.03",
                        "im2im_epoch": 44,
                        "unet_im2im_root": "unet_im2im_epochs50_bs4_2025-06-29-21.40.35",
                        "unet_im2im_epoch": 10,
                        "quantile_root": "unet_quantile_epochs50_bs4_2025-06-29-20.06.00",
                        "quantile_epoch": 35,
                        "min_noise": 50,
                        "max_noise": 100,
                        "num_noise": 40,
                        "calib_data_path": ROOT + "/Denoising_FMD/calibration",
                        "test_data_path": ROOT + "/Denoising_FMD/validation",
                        "in_channels": 1,
                        "batch_size": 20,
                        "experiments_folder": ROOT + "/experiments"},              
            "real_noise_mice": {"experiment_type": "real_noise",
                                "im2im_root": "im2im_epochs100_bs4_2025-07-05-22.51.09_mice",
                                "im2im_epoch": 35,
                          "unet_im2im_root": "unet_im2im_epochs100_bs4_2025-06-26-20.30.31_mice",
                          "unet_im2im_epoch": 5,
                          "quantile_root": "unet_quantile_epochs100_bs4_2025-06-26-20.30.31_mice",
                          "quantile_epoch": 10,
                          "calib_data_path": ROOT + "/Denoising_FMD/real_noise_new/calib_Confocal_MICE/calib_MICE",
                          "test_data_path": ROOT + "/Denoising_FMD/real_noise_new/calib_Confocal_MICE/val_MICE",
                          "in_channels": 1,
                          "batch_size": 20,
                          "experiments_folder": ROOT + "/experiments"},
            # "real_noise_bpae": {"experiment_type": "real_noise",
            #               "im2im_root": "im2im_epochs100_bs4_2025-07-05-22.51.09_bpae",
            #               "im2im_epoch": 35,
            #               "unet_im2im_root": "unet_im2im_epochs100_bs4_2025-06-26-20.30.31_bpae",
            #               "unet_im2im_epoch": 5,
            #               "quantile_root": "unet_quantile_epochs100_bs4_2025-06-26-20.30.31_bpae",
            #               "quantile_epoch": 10,
            #               "calib_data_path": ROOT + "/Denoising_FMD/real_noise_new/calib_Confocal_BPAE_G/calib_BPAE_G",
            #               "test_data_path": ROOT + "/Denoising_FMD/real_noise_new/calib_Confocal_BPAE_G/val_BPAE_G",
            #               "in_channels": 1,
            #               "batch_size": 20,
            #               "experiments_folder": ROOT + "/experiments"},
             "mri": {"experiment_type": "mri",
                     "im2im_root": "im2im_epochs50_bs32_2025-07-05-22.51.39",
                     "im2im_epoch": 35,
                     "unet_im2im_root": "unet_im2im_epochs50_bs12_2025-06-26-23.16.13",
                     "unet_im2im_epoch": 25,
                     "quantile_root": "unet_quantile_epochs50_bs12_2025-06-26-22.00.04",
                     "quantile_epoch": 30,
                     "calib_data_path": ROOT + "/Fast_MRI/RAW_singlecoil_train/calibration",
                     "test_data_path": ROOT + "/Fast_MRI/RAW_singlecoil_train/validation",
                     "in_channels": 1,
                     "batch_size": 32,
                     "experiments_folder": ROOT + "/experiments"},
             "qpi": {"experiment_type": "qpi",
                     "im2im_root": "im2im_epochs50_bs72_2025-09-11-14.29.04",
                     "im2im_epoch": 15,
                     "unet_im2im_root": "unet_im2im_epochs50_bs72_2025-08-26-11.10.34",
                     "unet_im2im_epoch": 15,
                     "quantile_root": "unet_quantile_epochs50_bs72_2025-08-26-11.10.45",
                     "quantile_epoch": 20,
                     "data_root": ROOT + "/bsccm",
                     "in_channels": 2,
                     "batch_size": 16,
                     "experiments_folder": ROOT + "/experiments",
                     "end_epoch": 25},
            "CT": {"experiment_type": "CT",
                    "im2im_root": "im2im_epochs50_bs4_2025-09-17-10.49.16",
                    "im2im_epoch": 35,
                    "quantile_root": "unet_quantile_epochs50_bs4_2025-09-17-10.47.41",
                     "quantile_epoch": 35,
                     "unet_im2im_root": "unet_im2im_epochs50_bs4_2025-09-17-10.49.15",
                     "unet_im2im_epoch": 20,
                     "data_root": ROOT + "/CT_Scan",
                     "in_channels": 1,
                     "batch_size": 16,
                     "experiments_folder": ROOT + "/experiments",
                     "end_epoch": 50}
            #  "motionblur": {"experiment_type": "motionblur",
            #               "im2im_root": "im2im_epochs50_bs24_2025-06-03-16.40.40", # 12 kernel size only
            #               "im2im_epoch": 50,
            #               "unet_im2im_root": "unet_im2im_epochs50_bs16_2025-06-26-14.41.10",
            #               "unet_im2im_epoch": 50,
            #               "quantile_root": "unet_quantile_epochs50_bs24_2025-06-26-03.47.23",
            #               "quantile_epoch": 50,
            #               "kernel_min": 12,
            #               "kernel_max": 12,
            #               "intensity": 0.1,
            #               "n_blur_samples": 1,
            #               "data_root": ROOT + "/motionblur",
            #               "in_channels": 1,
            #               "batch_size": 24,
            #               "experiments_folder": ROOT + "/experiments",
            #               "end_epoch": 50,},
            }

def run(tasks: List[str] = None,
        calib_subset: int = 2000,
        test_subset: int = 2000,
        intervals_subset: int = 500_000,
        output_dir: str = None,
        device: torch.device = None,
        use_submitit: bool = True,
        partition: str = "monakhova") -> Tuple[Dict, Dict, pd.DataFrame]:
    print(f"Device {device}")

    save_dir = Path(output_dir) if output_dir else Path(f"uqnet_results_{time.strftime('%Y-%m-%d-%H.%M')}")
    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Final results will be saved in: {save_dir}")

    if tasks is None:
        tasks = list(BEST_RUNS.keys())

    summary, all_stratified, all_df = {}, {}, []
    for task in tasks:
        print(f"\n--- Starting Experiment: {task} ---")
        if task not in BEST_RUNS:
            raise ValueError(f"Task '{task}' not found in BEST_RUNS. Available: {list(BEST_RUNS.keys())}")
        
        info = BEST_RUNS[task]
        experiment_type = info['experiment_type']
        calib_results: List[Dict[str, Any]]
        if use_submitit:
            calibration_manager = CalibrationManager(experiment_type, info, 
                                                     calib_subset=calib_subset, 
                                                     partition=partition,
                                                     ignore_checkpoints=False)
            requests = [
                ("im2im", info['im2im_epoch']),
                ("unet_im2im", info['unet_im2im_epoch']),
                ("quantile", info['quantile_epoch'])
            ]
            calib_results = calibration_manager.calibrate_multiple(requests)
        else:
            _, im2im_lambda = calibrate_im2im(experiment_type, info, device, calib_subset)
            _, im2im_deep_lambda = calibrate_im2im_deep(experiment_type, info, device, calib_subset)
            _, lower_q, upper_q = calibrate_quantile(experiment_type, info, device, calib_subset)
            calib_results = [
                {'net_type': 'im2im', 'lambda': im2im_lambda, 'epoch': info['im2im_epoch']},
                {'net_type': 'unet_im2im', 'lambda': im2im_deep_lambda, 'epoch': info['unet_im2im_epoch']},
                {'net_type': 'quantile', 'lower_q': lower_q, 'upper_q': upper_q, 'epoch': info['quantile_epoch']},
            ]
        # calib_risk = get_calibration_risk(experiment_type, info, device, calib_subset)
        # violations, total = check_crossings(experiment_type, info, device, calib_subset)

        analysis_manager = AnalysisManager(experiment_type, info, test_subset, intervals_subset, 
                                           partition, ignore_checkpoints=False)
        analysis_results = analysis_manager.analyze_multiple(calib_results)

        dfs_list = []
        risk_dict = {}
        for result in analysis_results:
            net_type = result['net_type']
            model_key = "im2im_deep" if net_type == "unet_im2im" else net_type

            df = result['df']
            df['method'] = model_key
            dfs_list.append(df)
            risk_dict[model_key] = result['risk']
        
        df_intervals_missed = pd.concat(dfs_list, ignore_index=True, copy=False)
        df_intervals_missed["experiment"] = task
        df_intervals_missed["experiment"] = df_intervals_missed["experiment"].astype("category")
        all_df.append(df_intervals_missed)
        
        stratified_results = compute_stratified_risk(df_intervals_missed)
        # for method in ['im2im', 'im2im_deep', 'quantile']:
        for method in ['im2im_deep', 'quantile']:
            print(f"{method} Stratified Risk")
            print(json.dumps(_prepare_for_json(stratified_results[method]), indent=4))
            print(f"{method} Average Risk: {risk_dict[method]['mean_risk']:.4f}")
        all_stratified[task] = stratified_results
        
        summary[task] = {
            # "calib_risk": calib_risk,
            # "crossings_violations": violations,
            # "crossings_total": total,
            "risk": risk_dict,
            "mean_interval_length": {
                "im2im": df_intervals_missed[df_intervals_missed["method"] == "im2im"]["interval"].mean(),
                "im2im_deep": df_intervals_missed[df_intervals_missed["method"] == "im2im_deep"]["interval"].mean(),
                "quantile": df_intervals_missed[df_intervals_missed["method"] == "quantile"]["interval"].mean(),
            },
            "std_interval_length": {
                "im2im": df_intervals_missed[df_intervals_missed["method"] == "im2im"]["interval"].std(),
                "im2im_deep": df_intervals_missed[df_intervals_missed["method"] == "im2im_deep"]["interval"].std(),
                "quantile": df_intervals_missed[df_intervals_missed["method"] == "quantile"]["interval"].std(),
            },
        }

        if torch.cuda.is_available(): torch.cuda.empty_cache()
        print_memory_stats()
    
    if all_df:
        all_df = pd.concat(all_df, ignore_index=True, copy=False)
        all_df.to_feather(save_dir / "all_df.feather", compression="zstd")

        with open(save_dir / "summary.json", "w") as f:
            json.dump(_prepare_for_json(summary), f, indent=4, default=json_converter)
        with open(save_dir / "all_stratified.json", "w") as f:
            json.dump(_prepare_for_json(all_stratified), f, indent=4, default=json_converter)
        
        if len(summary) > 1:
            create_stratified_risk_plot(all_stratified, save_dir)
            create_risk_plot(summary, save_dir)
            create_violin_plot(all_df, summary, save_dir)
        
        print(f"\nAll experiments processed. Final aggregated results saved in: {save_dir}")
    else:
        print("No data to save. All DataFrames were empty.")
    
    return summary, all_stratified, all_df

def manage_checkpoint(checkpoint_file: Path, compute_func: Callable, *args: Any, **kwargs: Any) -> Any:
    """
    Manages loading from or computing and saving to a checkpoint file.
    Supported extensions: .pt, .json, .feather.
    """
    file_type = checkpoint_file.suffix
    load_device = kwargs.pop('map_location', None) # For torch.load

    if checkpoint_file.exists():
        print(f"Loading from checkpoint: {checkpoint_file}")
        if file_type == ".pt":
            return torch.load(checkpoint_file, map_location=load_device)
        elif file_type == ".json":
            with open(checkpoint_file, "r") as f:
                return json.load(f)
        elif file_type == ".feather":
            return pd.read_feather(checkpoint_file)
        else:
            raise ValueError(f"Unsupported file type for checkpoint: {file_type}")
    else:
        # Remove map_location if it was passed for torch.load, as compute_func won't expect it
        # (unless it's a specific kwarg for compute_func, handled by its signature)
        if 'map_location' in compute_func.__code__.co_varnames and load_device:
             kwargs['map_location'] = load_device

        result = compute_func(*args, **kwargs)
        
        checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving checkpoint to: {checkpoint_file}")
        if file_type == ".pt":
            torch.save(result, checkpoint_file)
        elif file_type == ".json":
            serializable_result = _prepare_for_json(result)
            with open(checkpoint_file, "w") as f:
                json.dump(serializable_result, f, indent=4)
        elif file_type == ".feather":
            if not isinstance(result, pd.DataFrame):
                raise ValueError(f"Result for .feather checkpoint must be a pandas DataFrame, got {type(result)}")
            result.to_feather(checkpoint_file, compression="zstd")
        else:
            raise ValueError(f"Unsupported file type for checkpoint: {file_type}")
        return result
    
def calibrate_im2im(experiment_type: str, info: Dict, device: torch.device, 
                    calib_subset: int = 2000, epoch=None) -> Tuple[nn.Module, float]:
    epoch = epoch if epoch is not None else info["im2im_epoch"]
    im2im_model, run_folder = load_checkpoint_for_inference(net="im2im", in_channels=info["in_channels"], 
                                     experiment_type=experiment_type, run_folder_root=info["im2im_root"], 
                                     epoch=epoch, device=device, experiments_folder=info["experiments_folder"])
    calib_dataloader = get_calib_dataloader(experiment_type, info, net='im2im', 
                                            batch_size_multiplier=1.5, calib_subset=calib_subset)
    
    analysis_dir = Path(run_folder) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    alpha, min_l, max_l, num_l = 0.1, 0, 2, 300
    im2im_lambda_file = analysis_dir / f"im2im_epoch{epoch}_lambda.pt"
    im2im_lambda = manage_checkpoint(im2im_lambda_file, compute_optimal_lambdas,
                                     dataloader=calib_dataloader, model=im2im_model, alpha=alpha,
                                     min_lam=min_l, max_lam=max_l, num_lam=num_l, device=device)
    del calib_dataloader
    return im2im_model, im2im_lambda

def calibrate_im2im_deep(experiment_type: str, info: Dict, device: torch.device, 
                         calib_subset: int = 2000, epoch=None) -> Tuple[nn.Module, float]:
    epoch = epoch if epoch is not None else info["unet_im2im_epoch"]
    im2im_deep_model, run_folder = load_checkpoint_for_inference(net="unet_im2im", in_channels=info["in_channels"],
                                           experiment_type=experiment_type, run_folder_root=info["unet_im2im_root"],
                                           epoch=epoch, device=device, experiments_folder=info["experiments_folder"])
    calib_dataloader = get_calib_dataloader(experiment_type, info, net="unet_im2im", 
                                            batch_size_multiplier=1.5, calib_subset=calib_subset)

    analysis_dir = Path(run_folder) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    alpha, min_l, max_l, num_l = 0.1, 0, 2, 300
    im2im_deep_lambda_file = analysis_dir / f"unet_im2im_epoch{epoch}_lambda.pt"
    im2im_deep_lambda = manage_checkpoint(im2im_deep_lambda_file, compute_optimal_lambdas,
                                          dataloader=calib_dataloader, model=im2im_deep_model, alpha=alpha,
                                          min_lam=min_l, max_lam=max_l, num_lam=num_l, device=device)
    del calib_dataloader
    return im2im_deep_model, im2im_deep_lambda

def calibrate_quantile(experiment_type: str, info: Dict, device: torch.device, 
                       calib_subset: int = 2000, epoch=None) -> Tuple[nn.Module, float, float]:
    epoch = epoch if epoch is not None else info["quantile_epoch"]
    quantile_model, run_folder = load_checkpoint_for_inference(net="unet_quantile", in_channels=info["in_channels"],
                                        experiment_type=experiment_type, run_folder_root=info["quantile_root"],
                                        epoch=epoch, device=device, 
                                        experiments_folder=info["experiments_folder"])
    calib_dataloader = get_calib_dataloader(experiment_type, info, net="quantile", calib_subset=calib_subset)

    analysis_dir = Path(run_folder) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    target_alpha = 0.1
    lower_min, lower_max = 0, 0.1
    upper_min, upper_max = 0.9, 1
    quantile_qs_file = analysis_dir / f"quantile_qs_epoch{epoch}.json"
    quantile_qs = manage_checkpoint(quantile_qs_file, compute_optimal_lambdas_quantile,
                                     dataloader=calib_dataloader, model=quantile_model, alpha=target_alpha,
                                     lower_min=lower_min, lower_max=lower_max, 
                                     upper_min=upper_min, upper_max=upper_max,
                                     max_iterations=20, device=device)
    
    del calib_dataloader
    return quantile_model, quantile_qs["lower_q"], quantile_qs["upper_q"]

def get_calibration_risk(experiment_type: str, info: Dict, device: torch.device, 
                         calib_subset: int = 2000) -> Dict:
    im2im_model, im2im_folder = calibrate_im2im(experiment_type, info, device, calib_subset)
    im2im_deep_model, im2im_deep_folder = calibrate_im2im_deep(experiment_type, info, device, calib_subset)
    quantile_model, quantile_folder = calibrate_quantile(experiment_type, info, device, calib_subset)

    im2im_lambda = torch.load(im2im_folder / "analysis" / f"im2im_epoch{info['im2im_epoch']}_lambda.pt")
    im2im_deep_lambda = torch.load(im2im_deep_folder / "analysis" / f"unet_im2im_epoch{info['unet_im2im_epoch']}_lambda.pt")
    with open(quantile_folder / "analysis" / f"quantile_qs_epoch{info['quantile_epoch']}.json", 'r') as f:
        q_data = json.load(f)
        lower_q, upper_q = q_data['lower_q'], q_data['upper_q']

    calib_dataloader = get_calib_dataloader(experiment_type, info, net="quantile", 
                                            batch_size_multiplier=0.5, calib_subset=calib_subset)
    calib_risk = get_risk(dataloader=calib_dataloader, im2im_model=im2im_model, im2im_lambda=im2im_lambda,
                          im2im_deep_model=im2im_deep_model, im2im_deep_lambda=im2im_deep_lambda,
                          quantile_model=quantile_model, lower_q=lower_q, upper_q=upper_q, device=device)
    print("Calibration risk: ", json.dumps(_prepare_for_json(calib_risk), indent=4))
    del calib_dataloader
    return calib_risk

def check_crossings(experiment_type: str, info: Dict, device: torch.device, 
                    calib_subset: int = 2000) -> Tuple[int, int]:
    epoch = info["quantile_epoch"]
    quantile_model, run_folder = load_checkpoint_for_inference(
        net="unet_quantile", in_channels=info["in_channels"],
        experiment_type=experiment_type, run_folder_root=info["quantile_root"],
        epoch=epoch, device=device, experiments_folder=info["experiments_folder"])
    
    analysis_dir = run_folder / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    crossings_dataloader = get_calib_dataloader(experiment_type, info, net="quantile",
                                                 batch_size_multiplier=0.1, calib_subset=calib_subset)
    quantiles = torch.arange(0.1, 1.0, 0.1).to(device)
    crossings_file = analysis_dir / "quantile_crossings.json"
    crossings_data = manage_checkpoint(crossings_file, check_quantile_crossings,
                                        dataloader=crossings_dataloader, model=quantile_model, 
                                        quantiles=quantiles, device=device)
    violations, total = crossings_data["violations"], crossings_data["total"]
    del crossings_dataloader
    return violations, total

def compute_stratified_risk(df_intervals_missed: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    stratified_results = {}

    for method in ["im2im", "im2im_deep", "quantile"]:
        method_df = df_intervals_missed[df_intervals_missed["method"] == method][["interval", "missed"]]
        stratified_results[method] = size_stratified_risk(method_df)

    return stratified_results

def create_stratified_risk_plot(all_stratified: Dict[str, Dict[str, Dict[str, float]]], save_dir: Path):
    categories  = ["Short", "Short-Medium", "Medium-Long", "Long"]
    experiments = list(all_stratified.keys())
    methods     = ["im2im_deep", "quantile"]

    xtick_labels = []
    for exp in experiments:
        xtick_labels.extend([
            f"{exp}\nIm2Im-Deep",
            f"{exp}\nQUTCC"
        ])

    data = {cat: [] for cat in categories}
    for exp in experiments:
        for m in methods:
            for cat in categories:
                data[cat].append(all_stratified[exp][m][cat])

    fig, ax = plt.subplots(figsize=(14, 6))
    bar_width = 0.18
    x = np.arange(len(xtick_labels))

    colors_im2im_deep = ['#e6e6ff', '#bfbfff', '#9999ff', '#7373ff']
    colors_quantile = ["#ffe6e6", '#ffbfbf', '#ff9999', '#ff7373']
    palette_map = {"im2im_deep": colors_im2im_deep, "quantile": colors_quantile}
    for i, cat in enumerate(categories):
        bar_colors = []
        for lbl in xtick_labels:
            method = "im2im_deep" if "Deep" in lbl else "quantile"
            bar_colors.append(palette_map[method][i])
        ax.bar(x + (i - 1.5) * bar_width,   
            data[cat],
            width=bar_width,
            label=cat,
            color=bar_colors,
            edgecolor='white',
            linewidth=0.5)

    ax.set_xticks(x, labels=xtick_labels, fontsize=10, ha="center")
    ax.tick_params(axis="y", labelsize=16)

    ax.set_ylim(0, 0.2)
    ax.spines[['top', 'right']].set_visible(False)
    ax.axhline(0.10, ls="--", color="gray", lw=1)
    ax.text(len(x)-0.3, 0.101, r"$\alpha$", va="bottom", ha="left", color="gray", fontsize=16)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    legend_handles = []
    for i, cat in enumerate(categories):
        g_rgb = to_rgb(colors_im2im_deep[i])
        r_rgb = to_rgb(colors_quantile[i])
        blended = to_hex([(g + r) / 2 for g, r in zip(g_rgb, r_rgb)])
        legend_handles.append(Patch(facecolor=blended, edgecolor='white',
                                    linewidth=0.5, label=cat))

    ax.legend(handles     = legend_handles,
              fontsize    = 16,
              frameon     = True,
              handlelength= 2.5,
              bbox_to_anchor=(1, 1),
              loc='upper right')
    
    fig.tight_layout()
    fig.savefig(save_dir / "stratified_risk_all_tasks.pdf", dpi=300, bbox_inches="tight")
    print("Saved stratified risk plot to:", save_dir / "stratified_risk_all_tasks.pdf")

def create_risk_plot(summary: Dict[str, Dict], save_dir: Path):
    experiments = list(summary.keys())
    methods     = ["im2im_deep", "quantile"]
    color_map   = {"im2im_deep": "#7373ff", "quantile": "#ff7373"}

    xtick_labels, risk_values, std, bar_colors = [], [], [], []
    for exp in experiments:
        for method in methods:
            risk_info = summary[exp]["risk"][method]
            n_missed, n_total, mean_risk, std_risk = risk_info["missed"], risk_info["n"], risk_info["mean_risk"], risk_info["std_risk"]
            # print(f"{exp} {method} - Calculated risk: {n_missed / n_total:.4f}, Mean Risk: {mean_risk:.4f}, Std Risk: {std_risk:.4f}")
            risk_values.append(mean_risk)
            std.append(std_risk / 2)
            if method == "im2im_deep": xtick_labels.append(f"{exp}\nIm2Im-Deep")
            else: xtick_labels.append(f"{exp}\nQUTCC")
            bar_colors.append(color_map[method])

    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.6

    x = np.arange(len(risk_values))
    ax.bar(x, risk_values, width=bar_width, color=bar_colors, edgecolor='white', linewidth=0.5, yerr=std, capsize=5)
    ax.set_xticks(x, labels=xtick_labels, fontsize=10, ha="center")
    ax.tick_params(axis="y", labelsize=16) 
    
    ax.set_ylim(0, 0.2)
    ax.spines[['top', 'right']].set_visible(False)
    ax.axhline(0.10, ls="--", color="gray", lw=1)
    ax.text(len(x)-0.3, 0.101, r"$\alpha$", va="bottom", ha="left", color="gray", fontsize=16)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    fig.tight_layout()
    fig.savefig(save_dir / "risk_all_tasks.pdf", dpi=300, bbox_inches="tight")
    print("Saved risk plot to:", save_dir / "risk_all_tasks.pdf")

def create_violin_plot(all_df: pd.DataFrame, summary: Dict[str, Dict], save_dir: Path):
    df_violin = all_df[(all_df["method"] == "im2im_deep") | 
                    (all_df["method"] == "quantile")][["experiment", "method", "interval"]]
    method_order = ["im2im_deep", "quantile"]
    method_labels = {"im2im_deep": "Im2Im-Deep", "quantile": "QUTCC"}
    palette = {'im2im_deep': '#7373ff', 'quantile': '#ff7373'}

    multiplier = 2.5
    fig, ax = plt.subplots(figsize=(5.52 * multiplier, 2.5 * multiplier))
    sns.violinplot(
        data         = df_violin,
        x            = "experiment",
        y            = "interval",
        hue          = "method",
        order        = list(summary.keys()),
        hue_order    = method_order,
        split        = True,
        density_norm = "count",
        cut          = 0,
        inner        = None,
        palette      = palette,
        ax           = ax
    )

    ax.set_xlabel("") # Method
    ax.set_ylabel("") # Interval Size
    ax.set_ylim(0, 0.35)

    legend_elements = [
        Patch(facecolor=palette['im2im_deep'], label='Im2Im-Deep'), 
        Patch(facecolor=palette['quantile'], label='QUTCC')
    ]
    ax.legend(handles=legend_elements, fontsize=8 * multiplier, loc='upper right', bbox_to_anchor=(1.2, 1.2))

    ax.tick_params(axis="x", which='both', bottom=False, top=False, labelbottom=False)
    ax.tick_params(axis="y", labelsize=8 * multiplier)
    ax.spines[['top', 'right']].set_visible(False)    
    fig.tight_layout()

    violin_plot_path = save_dir / "interval_length_split_violin_all_tasks.png"
    fig.savefig(violin_plot_path, dpi=300, bbox_inches="tight")
    print("Saved split violin plot to:", violin_plot_path)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    summary, all_stratified, all_df = run(
        # can replace with experiment types like ["gaussian", "mri", "qpi"]
        tasks=None, 
        calib_subset=2000,
        test_subset=2000,
        intervals_subset=500_000,
        device=device,
        use_submitit=True,
        partition="gpu",
    )