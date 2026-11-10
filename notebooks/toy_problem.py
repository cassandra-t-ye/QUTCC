"""Library code for the QUTCC Gaussian toy problem (Part 1 of visualize_checkpoint.ipynb).

Holds everything that isn't specific to a single notebook run: the toy
inverse-problem config and construction, downloading/loading the three
pretrained checkpoints (QUTCC quantile model, MC-Dropout, Deep Ensemble)
from Google Drive, and the inference/PDF-recovery utilities used to compare
them against the closed-form posterior. The notebook itself only wires
these together: build the problem, load the weights, run inference, plot.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from matplotlib.gridspec import GridSpecFromSubplotSpec
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.stats import norm
from sklearn.isotonic import IsotonicRegression
from torch.utils.data import DataLoader

from models import create_model2
from models.dropout import DropoutUNet

# --- Pretrained checkpoints, hosted so nobody has to retrain from scratch ---
GDRIVE_CHECKPOINTS_URL = "https://drive.google.com/drive/folders/1tHtEUm6lk3vjL3cBXgzIEwn_1FDRvCwi?usp=sharing"
REQUIRED_CHECKPOINTS = ["quantile.pth", "mc_dropout.pth", "ensemble.pth"]
# Fixed relative to this file (notebooks/toy_problem.py), so it resolves the same
# way regardless of the notebook's working directory -- no user edit needed.
DEFAULT_CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "toy_checkpoints")

# --- Toy inverse problem (linear-Gaussian measurement of an MNIST digit) ---
N = 32                  # image side length
N_MEAS = 512            # number of linear measurements
NOISE_STD = 0.1         # measurement noise std
GT_IDX = 32             # test-set index of the ground-truth digit (a "3")
PRIOR_DIGIT = 3         # MNIST class used to estimate the Gaussian prior
SEED = 0

# --- Inference ---
NUM_MC_SAMPLES = 50     # stochastic forward passes at inference for MC-Dropout
K = 101                 # number of quantile levels queried for QUTCC

# Shared U-Net backbone -- identical for all three methods (same size/complexity);
# only the output head width and dropout rate differ per method.
BACKBONE_KWARGS = dict(
    image_size=N,
    num_channels=128,           # must be divisible by 32 (hardcoded GroupNorm(32, ...) in models/nn.py)
    num_res_blocks=3,
    channel_mult="1,2",         # single downsample 32 -> 16
    attention_resolutions="16", # fires once, cheap
    num_heads=1,
    use_scale_shift_norm=False,
    resblock_updown=False,
)


# ============================= Toy problem =================================

def forward_model(x, A):
    return (x.reshape(-1, N ** 2) @ A.T).reshape(-1, A.shape[0])


def get_gaussian_posterior(obs, A, cov_llh, mean_prior, cov_prior):
    Lam = A.T @ torch.inverse(cov_llh) @ A + torch.inverse(cov_prior)
    cov_pos = torch.inverse(Lam)
    mean_pos = cov_pos @ (A.T @ torch.inverse(cov_llh) @ obs.flatten() + torch.inverse(cov_prior) @ mean_prior.flatten())
    return mean_pos, cov_pos


def build_toy_problem(data_root, seed=SEED):
    """Builds a linear-Gaussian measurement of an MNIST digit with a closed-form
    Gaussian posterior -- the ground truth all three methods are compared against."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Resize((N, N)),
    ])
    train_set = torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform=transforms)
    test_set = torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform=transforms)
    train_loader = DataLoader(train_set, batch_size=10000, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=50, shuffle=False)
    imgs_train, digits_train = next(iter(train_loader))
    imgs, digits = next(iter(test_loader))

    gt = imgs[GT_IDX].squeeze(0)
    print(f"Ground truth digit label: {digits[GT_IDX].item()}")

    A = torch.randn(N_MEAS, N ** 2) / N
    obs = forward_model(gt, A)
    obs = obs + torch.randn_like(obs) * NOISE_STD
    cov_llh = torch.eye(N_MEAS) * NOISE_STD ** 2

    train_imgs = imgs_train[digits_train == PRIOR_DIGIT].clone()
    train_imgs += torch.randn_like(train_imgs) * 0.05
    mu_prior = train_imgs.mean(0).squeeze()
    cov_prior_full = train_imgs.reshape(-1, N ** 2).permute(1, 0).cov(correction=0)
    sigma_prior = cov_prior_full.diag().view(N, N).sqrt()
    cov_prior = torch.diag(sigma_prior.flatten() ** 2) / 4

    mean_pos, cov_pos = get_gaussian_posterior(obs, A, cov_llh, mu_prior, cov_prior)
    mean_pos = mean_pos.view(N, N)
    sigma_pos = cov_pos.diag().view(N, N).sqrt()

    return dict(
        gt=gt, A=A, obs=obs, cov_llh=cov_llh,
        mu_prior=mu_prior, sigma_prior=sigma_prior,
        mean_pos=mean_pos, sigma_pos=sigma_pos,
    )


# ======================= Pretrained checkpoint loading =======================

def download_pretrained_checkpoints(checkpoint_dir, folder_url=GDRIVE_CHECKPOINTS_URL):
    """Pulls the three pretrained toy-problem checkpoints from Google Drive into
    checkpoint_dir, if any of them are missing there."""
    missing = [f for f in REQUIRED_CHECKPOINTS if not os.path.exists(os.path.join(checkpoint_dir, f))]
    if not missing:
        print(f"All checkpoints already present in {checkpoint_dir}.")
        return

    import gdown  # raises a clear ImportError if missing (pip install gdown)

    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Missing checkpoints {missing} -- downloading pretrained weights from Google Drive...")
    gdown.download_folder(url=folder_url, output=checkpoint_dir, quiet=False, use_cookies=False)


def load_pretrained_models(checkpoint_dir, device):
    """Loads the QUTCC quantile model, MC-Dropout model, and Deep Ensemble members
    from checkpoint_dir (download_pretrained_checkpoints first if they're not there yet).

    Returns (quantile_model, mc_model, ensemble_members): mc_model is wrapped in
    QUTCC's DropoutUNet helper, ensemble_members is a list of models.
    """
    missing = [f for f in REQUIRED_CHECKPOINTS if not os.path.exists(os.path.join(checkpoint_dir, f))]
    if missing:
        raise FileNotFoundError(
            f"Missing checkpoints {missing} in {checkpoint_dir} -- call "
            f"download_pretrained_checkpoints({checkpoint_dir!r}) first."
        )

    quantile_model = create_model2(**BACKBONE_KWARGS, dropout=0, num_measurements=1, out_channels=1).to(device)
    quantile_model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "quantile.pth"), map_location=device))
    quantile_model.eval()
    print(f"[quantile] parameters: {sum(p.numel() for p in quantile_model.parameters()):,}")

    mc_base_model = create_model2(**BACKBONE_KWARGS, dropout=0.1, num_measurements=1, out_channels=1).to(device)
    mc_base_model.load_state_dict(torch.load(os.path.join(checkpoint_dir, "mc_dropout.pth"), map_location=device))
    mc_base_model.eval()
    print(f"[mc_dropout] parameters: {sum(p.numel() for p in mc_base_model.parameters()):,}")
    # Wrap in QUTCC's own MC-Dropout helper: forward(..., reduce=False) keeps
    # dropout active and returns the stack of stochastic samples.
    mc_model = DropoutUNet(mc_base_model)

    state_dicts = torch.load(os.path.join(checkpoint_dir, "ensemble.pth"), map_location=device)
    ensemble_members = []
    for i, state_dict in enumerate(state_dicts):
        member = create_model2(**BACKBONE_KWARGS, dropout=0, num_measurements=1, out_channels=2).to(device)
        member.load_state_dict(state_dict)
        member.eval()
        if i == 0:
            print(f"[ensemble] parameters per member: {sum(p.numel() for p in member.parameters()):,}")
        ensemble_members.append(member)

    return quantile_model, mc_model, ensemble_members


# ========================= Inference / PDF recovery ===========================

def quantile_to_pdf(qf, levels, sigma=2, step=3):
    """Numerically differentiate a (possibly noisy) quantile function to get a PDF."""
    step = max(1, min(step, len(qf) - 1))
    qf_smooth = gaussian_filter1d(qf, sigma=sigma, mode="nearest")
    dq = qf_smooth[step:] - qf_smooth[:-step]
    dt = levels[step:] - levels[:-step]
    slope = dq / dt

    good = slope > 1e-5
    x = 0.5 * (qf_smooth[step:] + qf_smooth[:-step])
    pdf = np.zeros_like(slope)
    pdf[good] = 1.0 / slope[good]
    x, pdf = x[good], pdf[good]

    sort_idx = np.argsort(x)
    x, pdf = x[sort_idx], pdf[sort_idx]
    norm_const = np.trapz(pdf, x)
    if norm_const > 0:
        pdf = pdf / norm_const
    return x, pdf


def get_quantile_functions(model, x_adj_test, levels, device):
    q_tensor = torch.tensor(levels, device=device)
    x_adj_batch = x_adj_test.repeat(len(levels), 1, 1, 1)

    with torch.no_grad():
        qf = model(x_adj_batch, q_tensor).squeeze(1).cpu().numpy()  # (K, N, N)

    n_crossings_before = int(np.sum(np.diff(qf, axis=0) < 0))
    print(f"[quantile] crossings before isotonic regression: {n_crossings_before}")

    # Enforce monotonicity pixel-wise (see analysis.py:check_crossings -- QUTCC's
    # learned quantile function is monotonic almost everywhere, but numerical
    # differentiation for the PDF is sensitive to the rare violation).
    ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
    qf_iso = np.empty_like(qf)
    for i in range(N):
        for j in range(N):
            qf_iso[:, i, j] = ir.fit_transform(levels, qf[:, i, j])

    n_crossings_after = int(np.sum(np.diff(qf_iso, axis=0) < 0))
    print(f"[quantile] crossings after  isotonic regression: {n_crossings_after}")
    return qf_iso


def get_mc_dropout_moments(mc_model, x_adj_test, device, num_samples=NUM_MC_SAMPLES):
    timevect = torch.full((1,), 0.5, device=device, dtype=torch.float32)
    with torch.no_grad():
        samples = mc_model(x_adj_test, timevect, num_samples=num_samples, reduce=False)  # (S, 1, 1, N, N)
    mean_map = samples.mean(dim=0).squeeze().cpu().numpy()
    var_map = samples.var(dim=0, unbiased=False).squeeze().cpu().numpy()
    return mean_map, var_map


def get_ensemble_moments(members, x_adj_test, device):
    """Combine member predictions via the law of total expectation/variance."""
    timevect = torch.full((1,), 0.5, device=device, dtype=torch.float32)
    means, variances = [], []
    with torch.no_grad():
        for member in members:
            raw_output = member(x_adj_test, timevect)
            means.append(raw_output[:, :1])
            variances.append(F.softplus(raw_output[:, 1:2]) + 1e-6)
    means = torch.stack(means, dim=0)       # (M, 1, 1, N, N)
    variances = torch.stack(variances, dim=0)

    ensemble_mean = means.mean(dim=0)
    ensemble_second_moment = (variances + means.square()).mean(dim=0)
    ensemble_variance = torch.clamp(ensemble_second_moment - ensemble_mean.square(), min=0.0)
    return ensemble_mean.squeeze().cpu().numpy(), ensemble_variance.squeeze().cpu().numpy()


def quantile_pixel_pdf_fn(qf, levels):
    def fn(i, j):
        return quantile_to_pdf(qf[:, i, j], levels)
    return fn


def gaussian_pixel_pdf_fn(mean_map, var_map):
    def fn(i, j):
        mu, sig = mean_map[i, j], max(np.sqrt(max(var_map[i, j], 0.0)), 1e-6)
        grid = np.linspace(mu - 6 * sig, mu + 6 * sig, 500)
        pdf = norm.pdf(grid, loc=mu, scale=sig)
        return grid, pdf
    return fn


def quantile_mean_std_maps(qf, levels):
    """Numerically integrate each pixel's QUTCC pdf to get mean/std maps."""
    mean_map = np.zeros((N, N))
    std_map = np.zeros((N, N))
    fn = quantile_pixel_pdf_fn(qf, levels)
    for i in range(N):
        for j in range(N):
            x, pdf = fn(i, j)
            m = np.trapz(pdf * x, x)
            var = np.trapz(pdf * (x - m) ** 2, x)
            mean_map[i, j] = m
            std_map[i, j] = np.sqrt(max(var, 0.0))
    return mean_map, std_map


def compare_to_true_gaussian(pixel_pdf_fn, mean_pos_np, sigma_pos_np):
    """Per-pixel KL(true || model), used only to pick illustrative pixels."""
    from scipy.special import rel_entr
    kl_map = np.zeros((N, N))
    eps = 1e-10
    for i in range(N):
        for j in range(N):
            x_q, pdf_q = pixel_pdf_fn(i, j)
            mu, sig = mean_pos_np[i, j], max(sigma_pos_np[i, j], 1e-6)
            grid_lo = min(x_q.min(), mu - 6 * sig)
            grid_hi = max(x_q.max(), mu + 6 * sig)
            grid = np.linspace(grid_lo, grid_hi, 500)
            pdf_q_grid = interp1d(x_q, pdf_q, bounds_error=False, fill_value=0.0)(grid)
            norm_const = np.trapz(pdf_q_grid, grid)
            if norm_const > 0:
                pdf_q_grid = pdf_q_grid / norm_const
            pdf_true_grid = norm.pdf(grid, loc=mu, scale=sig)
            kl_map[i, j] = np.trapz(rel_entr(pdf_true_grid + eps, pdf_q_grid + eps), grid)
    return kl_map


# ============================ Figure reproduction =============================

def prepare_figure_data(qf, levels, mc_mean, mc_var, ens_mean, ens_var, mean_pos_np, sigma_pos_np):
    """Computes QUTCC's mean/std maps and all three std maps, then picks two
    illustrative pixels (green = high-signal, cyan = low-signal) where QUTCC's
    recovered PDF best matches the true posterior. Returns a dict consumed by
    plot_comparison_figure."""
    print("Integrating QUTCC pdfs to get mean/std maps (this loops over all pixels)...")
    q_mean_map, q_std_map = quantile_mean_std_maps(qf, levels)
    mc_std_map = np.sqrt(np.maximum(mc_var, 0.0))
    ens_std_map = np.sqrt(np.maximum(ens_var, 0.0))

    print("Computing QUTCC KL-divergence map to pick well-matched pixels...")
    kl_map = compare_to_true_gaussian(quantile_pixel_pdf_fn(qf, levels), mean_pos_np, sigma_pos_np)

    # green = QUTCC's best-matched high-signal (top-quartile intensity) pixel
    high_signal_thresh = np.percentile(mean_pos_np, 75)
    kl_high_signal = np.where(mean_pos_np > high_signal_thresh, kl_map, np.inf)
    green_px = tuple(int(v) for v in np.unravel_index(np.argmin(kl_high_signal), kl_high_signal.shape))

    # cyan = QUTCC's best-matched low-signal / background (bottom-quartile intensity) pixel
    low_signal_thresh = np.percentile(mean_pos_np, 25)
    kl_low_signal = np.where(mean_pos_np < low_signal_thresh, kl_map, np.inf)
    cyan_px = tuple(int(v) for v in np.unravel_index(np.argmin(kl_low_signal), kl_low_signal.shape))

    print(f"green pixel: {green_px}  mean={mean_pos_np[green_px]:.4f}  sigma={sigma_pos_np[green_px]:.4f}")
    print(f"cyan  pixel: {cyan_px}  mean={mean_pos_np[cyan_px]:.4f}  sigma={sigma_pos_np[cyan_px]:.4f}")

    return dict(
        q_mean_map=q_mean_map, q_std_map=q_std_map,
        mc_std_map=mc_std_map, ens_std_map=ens_std_map,
        green_px=green_px, cyan_px=cyan_px,
    )


def plot_comparison_figure(qf, levels, mean_pos_np, sigma_pos_np, mc_mean, mc_var, ens_mean, ens_var, figure_data):
    """Reproduces the paper's toy-problem figure: mean/std maps for the true
    posterior and all three methods (left), plus per-pixel density overlays
    at the two illustrative pixels from prepare_figure_data (right)."""
    q_mean_map, q_std_map = figure_data["q_mean_map"], figure_data["q_std_map"]
    mc_std_map, ens_std_map = figure_data["mc_std_map"], figure_data["ens_std_map"]
    green_px, cyan_px = figure_data["green_px"], figure_data["cyan_px"]

    fig = plt.figure(figsize=(30, 6.5))
    gs = fig.add_gridspec(2, 7, width_ratios=[1, 1, 1, 1, 0.08, 0.5, 4.5], wspace=0.4, hspace=0.25)
    gs_right = GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[:, 6], hspace=0.55)

    mean_maps = [("True Posterior", mean_pos_np), ("QUTCC", q_mean_map),
                 ("Deep Ensemble", ens_mean), ("MC-Dropout", mc_mean)]
    std_maps = [("True Posterior", sigma_pos_np), ("QUTCC", q_std_map),
                ("Deep Ensemble", ens_std_map), ("MC-Dropout", mc_std_map)]

    vmin_mean, vmax_mean = mean_pos_np.min(), mean_pos_np.max()
    vmax_std = sigma_pos_np.max() * 1.3

    for col, (title, m) in enumerate(mean_maps):
        ax = fig.add_subplot(gs[0, col])
        im0 = ax.imshow(m, cmap="gray", vmin=vmin_mean, vmax=vmax_mean)
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        ax.plot(green_px[1], green_px[0], "o", color="lime", ms=8, mec="k", mew=0.5)
        ax.plot(cyan_px[1], cyan_px[0], "o", color="cyan", ms=8, mec="k", mew=0.5)
        if col == 0:
            ax.set_ylabel("Mean (μ)", fontsize=14, fontweight="bold")

    cax0 = fig.add_subplot(gs[0, 4])
    plt.colorbar(im0, cax=cax0)

    for col, (title, s) in enumerate(std_maps):
        ax = fig.add_subplot(gs[1, col])
        im1 = ax.imshow(s, cmap="gray", vmin=0, vmax=vmax_std)
        ax.set_xticks([]); ax.set_yticks([])
        ax.plot(green_px[1], green_px[0], "o", color="lime", ms=8, mec="k", mew=0.5)
        ax.plot(cyan_px[1], cyan_px[0], "o", color="cyan", ms=8, mec="k", mew=0.5)
        if col == 0:
            ax.set_ylabel("Std. Dev. (σ)", fontsize=14, fontweight="bold")

    cax1 = fig.add_subplot(gs[1, 4])
    plt.colorbar(im1, cax=cax1)

    quantile_fn = quantile_pixel_pdf_fn(qf, levels)
    mc_fn = gaussian_pixel_pdf_fn(mc_mean, mc_var)
    ens_fn = gaussian_pixel_pdf_fn(ens_mean, ens_var)

    rng = np.random.default_rng(0)
    n_samples = 20000
    hist_axes = []

    for row, (px, color, label) in enumerate([
        (green_px, "limegreen", f"Pixel {green_px}"),
        (cyan_px, "cyan", f"Pixel {cyan_px}"),
    ]):
        ax = fig.add_subplot(gs_right[row, 0])
        hist_axes.append(ax)
        i, j = px
        mu, sig = mean_pos_np[i, j], max(sigma_pos_np[i, j], 1e-6)

        true_samples = rng.normal(mu, sig, n_samples)
        ax.hist(true_samples, bins=100, density=True, color=color, alpha=0.6, label="True samples")

        grid = np.linspace(mu - 6 * sig, mu + 6 * sig, 500)
        ax.plot(grid, norm.pdf(grid, mu, sig), color="black", lw=2.5, label="Analytical")

        x_q, pdf_q = quantile_fn(i, j)
        x_e, pdf_e = ens_fn(i, j)
        x_m, pdf_m = mc_fn(i, j)
        ax.plot(x_q, pdf_q, color="tab:blue", lw=2, label="QUTCC")
        ax.plot(x_e, pdf_e, color="tab:red", lw=2, ls="--", label="Deep Ensemble")
        ax.plot(x_m, pdf_m, color="tab:orange", lw=2, ls=":", label="MC-Dropout")

        ax.axvline(mu, color="black", ls="--", lw=1.5)
        ax.set_xlim(-1.0, 1.5)
        ax.set_title(label, fontsize=13, fontweight="bold")
        ax.set_ylabel("Density", fontsize=12)
        if row == 1:
            ax.set_xlabel("Pixel value", fontsize=12)
        if row == 0:
            ax.legend(fontsize=10, loc="upper right")

    fig.align_ylabels(hist_axes)
    fig.suptitle("Gaussian toy example: True Posterior vs. QUTCC / Deep Ensemble / MC-Dropout",
                 fontsize=16, fontweight="bold", y=1.04)
    plt.show()
