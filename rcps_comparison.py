"""
Compare RCPS calibration against QUTCC's own calibration for the *same* quantile network.

Both methods start from the same heuristic interval - the network's 5th/95th percentile
predictions around its median prediction - and differ only in how they are calibrated:

  * QUTCC  : binary-search the quantile levels themselves (q_lo, q_hi) until each side's
             pixel-wise miss rate is <= alpha / 2, and query the net at those levels.
  * RCPS   : keep the quantile levels fixed at (0.05, 0.95) and scale the interval by a
             single scalar lambda, chosen as the smallest lambda whose Hoeffding-Bentkus
             upper confidence bound on the mean per-image miss fraction stays below alpha.

Both are then evaluated on the same test loader, so interval length and risk are directly
comparable.

Usage:
    python rcps_comparison.py --root /path/to/data --task mri
"""

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from evaluation import get_test_intervals_missed, size_stratified_risk
from evaluation.calibration import (compute_optimal_lambdas_quantile,
                                    compute_rcps_lambda_quantile)
from evaluation.calibration_utils import get_calib_dataloader, get_test_dataloader
from models.checkpointing import load_checkpoint_for_inference
from utils import _prepare_for_json

QUTCC_COLOR = "#2F6FBA"
RCPS_COLOR = "#E08214"


def load_best_runs(root: str) -> Dict[str, Any]:
    """Load best_runs.json and substitute the ROOT placeholder with the real data root."""
    with open(Path(__file__).parent / "best_runs.json", "r") as f:
        best_runs = json.load(f)

    def inject(d: Dict[str, Any]) -> None:
        for k, v in d.items():
            if isinstance(v, str) and v.startswith("ROOT/"):
                d[k] = v.replace("ROOT", root, 1)
            elif isinstance(v, dict):
                inject(v)

    inject(best_runs)
    return best_runs


def load_or_compute_qutcc_qs(calib_loader, model, alpha: float, analysis_dir: Path, save_dir: Path,
                             epoch: int, device: str, recalibrate: bool) -> Dict[str, float]:
    """
    QUTCC's own calibration. Reuses the cached quantile levels from the run's analysis dir when
    they exist; a recomputed result is written to `save_dir` so the original is never clobbered.
    """
    cache = analysis_dir / f"quantile_qs_epoch_{epoch}.json"
    if cache.exists() and not recalibrate:
        with open(cache, "r") as f:
            data = json.load(f)
        print(f"Using cached QUTCC calibration from {cache}: {data}")
        return {"lower_q": data["lower_q"], "upper_q": data["upper_q"]}

    data = compute_optimal_lambdas_quantile(
        dataloader=calib_loader, model=model, alpha=alpha,
        lower_min=1e-20, lower_max=0.5, upper_min=0.5, upper_max=1 - 1e-20,
        max_iterations=30, device=device,
    )
    with open(save_dir / f"quantile_qs_epoch_{epoch}.json", "w") as f:
        json.dump(data, f, indent=4)
    return data


def summarize(name: str, df: pd.DataFrame, risk: Dict[str, float], alpha: float) -> Dict[str, Any]:
    return {
        "method": name,
        "mean_interval_length": float(df["interval"].mean()),
        "median_interval_length": float(df["interval"].median()),
        "pixel_risk": float(risk["missed"] / risk["n"]),
        "mean_per_image_risk": float(risk["mean_risk"]),
        "std_per_image_risk": float(risk["std_risk"]),
        "alpha": alpha,
        "stratified_risk": size_stratified_risk(df),
    }


def make_plots(rcps_calib: Dict[str, Any], qutcc_summary: Dict[str, Any], rcps_summary: Dict[str, Any],
               df_qutcc: pd.DataFrame, df_rcps: pd.DataFrame, alpha: float, out_path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # --- (a) RCPS risk curve over lambda -------------------------------------------------
    ax = axes[0]
    lambdas = np.asarray(rcps_calib["lambdas"])
    risk_curve = np.asarray(rcps_calib["risk_curve"])
    ucb_curve = np.asarray(rcps_calib["risk_ucb_curve"])
    ax.plot(lambdas, risk_curve, lw=2, color=RCPS_COLOR, label=r"empirical risk $\hat{R}(\lambda)$")
    finite = ~np.isnan(ucb_curve)
    ax.plot(lambdas[finite], ucb_curve[finite], lw=2, ls="--", color="#7f7f7f",
            label=r"HB upper bound $\hat{R}^+(\lambda)$")
    ax.axhline(alpha, lw=1, color="#444444", ls=":", label=rf"$\alpha$ = {alpha:g}")
    ax.axvline(rcps_calib["lambda"], lw=1.5, color=RCPS_COLOR, alpha=0.6)
    ax.annotate(rf"$\hat{{\lambda}}$ = {rcps_calib['lambda']:.3f}",
                xy=(rcps_calib["lambda"], alpha), xytext=(6, 12), textcoords="offset points",
                color=RCPS_COLOR)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("risk (mean per-image miss fraction)")
    ax.set_title("(a) RCPS calibration curve")
    ax.set_ylim(0, min(1.0, alpha * 5))  # zoom on the region where the curves cross alpha
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, lw=0.6)

    # --- (b) interval length distributions ----------------------------------------------
    ax = axes[1]
    hi = float(max(df_qutcc["interval"].quantile(0.99), df_rcps["interval"].quantile(0.99)))
    bins = np.linspace(0, hi, 80)
    for df, label, color, summary in ((df_qutcc, "QUTCC", QUTCC_COLOR, qutcc_summary),
                                      (df_rcps, "RCPS", RCPS_COLOR, rcps_summary)):
        ax.hist(df["interval"], bins=bins, histtype="step", lw=2, color=color, density=True,
                label=f"{label} (mean {summary['mean_interval_length']:.4f})")
        ax.axvline(summary["mean_interval_length"], lw=1, ls="--", color=color, alpha=0.7)
    ax.set_xlabel("interval length")
    ax.set_ylabel("density")
    ax.set_title("(b) Interval length distribution")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, lw=0.6)

    # --- (c) size-stratified risk --------------------------------------------------------
    ax = axes[2]
    labels = ["Short", "Short-Medium", "Medium-Long", "Long"]
    x = np.arange(len(labels))
    width = 0.38
    for offset, summary, label, color in ((-width / 2, qutcc_summary, "QUTCC", QUTCC_COLOR),
                                          (width / 2, rcps_summary, "RCPS", RCPS_COLOR)):
        vals = [summary["stratified_risk"].get(k, 0.0) for k in labels]
        bars = ax.bar(x + offset, vals, width * 0.94, label=label, color=color)
        ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2, color="#333333")
    ax.axhline(alpha, lw=1, color="#444444", ls=":", label=rf"$\alpha$ = {alpha:g}")
    ax.set_xticks(x, labels, fontsize=9)
    ax.set_xlabel("interval-length quartile")
    ax.set_ylabel("risk")
    ax.set_title("(c) Size-stratified risk")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25, lw=0.6, axis="y")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved comparison figure to {out_path}")


def run(task: str, root: str, epoch: Optional[int], alpha: float, delta: float,
        q_lo: float, q_hi: float, q_mid: float, min_lam: float, max_lam: float, num_lam: int,
        use_ucb: bool, calib_subset: int, test_subset: int, intervals_subset: int,
        recalibrate: bool, output_dir: Optional[str], seed: int) -> Dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    best_runs = load_best_runs(root)
    if task not in best_runs:
        raise ValueError(f"Task '{task}' not found. Available: {list(best_runs.keys())}")
    info = best_runs[task]
    experiment_type = info["experiment_type"]
    epoch = epoch if epoch is not None else info["quantile_epoch"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Task={task} experiment={experiment_type} epoch={epoch} device={device}")

    model, run_folder = load_checkpoint_for_inference(
        net="unet_quantile", in_channels=info["in_channels"], experiment_type=experiment_type,
        run_folder_root=info["quantile_root"], epoch=epoch, device=device,
        experiments_folder=info["experiments_folder"],
    )
    analysis_dir = Path(run_folder) / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    save_dir = Path(output_dir) if output_dir else analysis_dir / "rcps_comparison"
    save_dir.mkdir(parents=True, exist_ok=True)

    # One calibration loader and one test loader, shared by both methods.
    calib_loader = get_calib_dataloader(experiment_type, info, net="quantile",
                                        batch_size_multiplier=0.9, calib_subset=calib_subset)
    test_loader = get_test_dataloader(experiment_type, info, net="quantile",
                                      batch_size_multiplier=0.7, test_subset=test_subset)

    # ---------------- calibrate ----------------
    print("\n" + "=" * 30 + " RCPS calibration " + "=" * 30)
    rcps_calib = compute_rcps_lambda_quantile(
        dataloader=calib_loader, model=model, alpha=alpha, delta=delta,
        q_lo=q_lo, q_mid=q_mid, q_hi=q_hi,
        min_lam=min_lam, max_lam=max_lam, num_lam=num_lam, use_ucb=use_ucb, device=device,
    )
    with open(save_dir / f"rcps_lambda_epoch{epoch}.json", "w") as f:
        json.dump(_prepare_for_json(rcps_calib), f, indent=4)

    print("\n" + "=" * 30 + " QUTCC calibration " + "=" * 30)
    qutcc_qs = load_or_compute_qutcc_qs(calib_loader, model, alpha, analysis_dir, save_dir,
                                        epoch, device, recalibrate)

    # ---------------- evaluate on the same test set ----------------
    print("\n" + "=" * 30 + " Test evaluation " + "=" * 30)
    df_qutcc, risk_qutcc = get_test_intervals_missed(
        "quantile", test_loader, model, qutcc_qs["lower_q"], qutcc_qs["upper_q"], device=device)
    df_rcps, risk_rcps = get_test_intervals_missed(
        "quantile_rcps", test_loader, model, rcps_calib["lambda"], None, device=device,
        q_lo=q_lo, q_mid=q_mid, q_hi=q_hi)

    if intervals_subset and len(df_qutcc) > intervals_subset:
        df_qutcc = df_qutcc.sample(n=intervals_subset, random_state=seed)
        df_rcps = df_rcps.sample(n=intervals_subset, random_state=seed)

    qutcc_summary = summarize("QUTCC", df_qutcc, risk_qutcc, alpha)
    qutcc_summary.update({"lower_q": qutcc_qs["lower_q"], "upper_q": qutcc_qs["upper_q"]})
    rcps_summary = summarize("RCPS", df_rcps, risk_rcps, alpha)
    rcps_summary.update({"lambda": rcps_calib["lambda"], "q_lo": q_lo, "q_hi": q_hi,
                         "calib_risk": rcps_calib["calib_risk"],
                         "calib_risk_ucb": rcps_calib["calib_risk_ucb"]})

    results = {
        "task": task, "experiment_type": experiment_type, "epoch": epoch,
        "alpha": alpha, "delta": delta, "seed": seed,
        "n_calib_images": rcps_calib["n_calib_images"],
        "n_test_pixels": int(risk_qutcc["n"]),
        "qutcc": qutcc_summary, "rcps": rcps_summary,
        "interval_length_ratio_rcps_over_qutcc": (rcps_summary["mean_interval_length"] /
                                                  qutcc_summary["mean_interval_length"]),
    }
    with open(save_dir / f"comparison_epoch{epoch}.json", "w") as f:
        json.dump(_prepare_for_json(results), f, indent=4)
    df_qutcc.to_feather(save_dir / f"intervals_qutcc_epoch{epoch}.feather", compression="zstd")
    df_rcps.to_feather(save_dir / f"intervals_rcps_epoch{epoch}.feather", compression="zstd")

    make_plots(rcps_calib, qutcc_summary, rcps_summary, df_qutcc, df_rcps, alpha,
               save_dir / f"rcps_vs_qutcc_{task}_epoch{epoch}.png")

    print_summary(results)
    return results


def print_summary(results: Dict[str, Any]) -> None:
    q, r = results["qutcc"], results["rcps"]
    print("\n" + "=" * 78)
    print(f"RCPS vs QUTCC - {results['task']} (epoch {results['epoch']}, "
          f"alpha={results['alpha']:g}, delta={results['delta']:g})")
    print("=" * 78)
    print(f"  QUTCC calibrated levels : q_lo={q['lower_q']:.5g}, q_hi={q['upper_q']:.5g}")
    print(f"  RCPS  calibrated lambda : {r['lambda']:.5g} "
          f"(on fixed q=({r['q_lo']}, {r['q_hi']}))")
    print()
    rows = [
        ("mean interval length", "mean_interval_length", "{:.5f}"),
        ("median interval length", "median_interval_length", "{:.5f}"),
        ("test risk (per pixel)", "pixel_risk", "{:.5f}"),
        ("test risk (per image, mean)", "mean_per_image_risk", "{:.5f}"),
        ("test risk (per image, std)", "std_per_image_risk", "{:.5f}"),
    ]
    print(f"  {'metric':<30}{'QUTCC':>14}{'RCPS':>14}{'RCPS / QUTCC':>16}")
    print("  " + "-" * 74)
    for label, key, fmt in rows:
        qv, rv = q[key], r[key]
        ratio = f"{rv / qv:.3f}" if qv else "n/a"
        print(f"  {label:<30}{fmt.format(qv):>14}{fmt.format(rv):>14}{ratio:>16}")
    print("\n  size-stratified risk (test):")
    print(f"  {'quartile':<30}{'QUTCC':>14}{'RCPS':>14}")
    print("  " + "-" * 58)
    for bin_label in ["Short", "Short-Medium", "Medium-Long", "Long"]:
        print(f"  {bin_label:<30}{q['stratified_risk'][bin_label]:>14.5f}"
              f"{r['stratified_risk'][bin_label]:>14.5f}")
    print("=" * 78 + "\n")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="mri", help="key in best_runs.json (default: mri)")
    p.add_argument("--root", required=True, help="data root that replaces the ROOT placeholder")
    p.add_argument("--epoch", type=int, default=None, help="checkpoint epoch (default: quantile_epoch)")
    p.add_argument("--alpha", type=float, default=0.1)
    p.add_argument("--delta", type=float, default=0.1, help="RCPS failure probability")
    p.add_argument("--q-lo", type=float, default=0.05)
    p.add_argument("--q-hi", type=float, default=0.95)
    p.add_argument("--q-mid", type=float, default=0.5)
    p.add_argument("--min-lam", type=float, default=0.0)
    p.add_argument("--max-lam", type=float, default=6.0)
    p.add_argument("--num-lam", type=int, default=1000)
    p.add_argument("--empirical-risk", action="store_true",
                   help="select lambda on the empirical risk instead of the HB upper bound")
    p.add_argument("--calib-subset", type=int, default=500)
    p.add_argument("--test-subset", type=int, default=1000)
    p.add_argument("--intervals-subset", type=int, default=500_000)
    p.add_argument("--recalibrate", action="store_true",
                   help="rerun QUTCC's quantile search instead of using the cached JSON")
    p.add_argument("--output-dir", default=None)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    run(task=a.task, root=a.root, epoch=a.epoch, alpha=a.alpha, delta=a.delta,
        q_lo=a.q_lo, q_hi=a.q_hi, q_mid=a.q_mid,
        min_lam=a.min_lam, max_lam=a.max_lam, num_lam=a.num_lam,
        use_ucb=not a.empirical_risk,
        calib_subset=a.calib_subset, test_subset=a.test_subset,
        intervals_subset=a.intervals_subset, recalibrate=a.recalibrate,
        output_dir=a.output_dir, seed=a.seed)
