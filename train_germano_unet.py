#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, random
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import sys
import traceback
import matplotlib.pyplot as plt
import numpy as np, torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from germano_common import *
from visualization_common import \
(
    save_field_centerline_comparison,
    save_field_error_images,
    save_field_images,
    save_field_spectrum_comparison,
)


def calculate_training_losses \
(
    model, coarse, filtered_target, dns_target, mean, std, dns_std,
    vorticity_std,
    alpha, lx, ly, lambda_dns, lambda_first_derivative,
    lambda_second_derivative, lambda_vorticity, vorticity_from_potential,
    inverse_cutoff_fraction, inverse_cutoff_order,
    inverse_filter_type, filtered_dns_shell_power,
    post_cutoff_fraction, post_cutoff_order, post_filter_type,
    inverse_method="exact", van_cittert_iterations=1,
    van_cittert_relaxation=1.0, lambda_kl=0.0,
    inverse_min_wavenumber=1.0,
    lambda_mse=0.1,
):
    """Return DNS L1 plus squared-error loss with optional vorticity and KL terms."""
    output_channels = 2 if vorticity_from_potential else 3
    prediction, kl = model(coarse, return_kl=True)
    predicted_filtered = \
    (
        prediction * std[:, :output_channels] + mean[:, :output_channels]
    )
    inverse_input = inverse_input_filter_torch \
    (
        predicted_filtered, lx, ly, inverse_filter_type,
        inverse_cutoff_fraction, inverse_cutoff_order,
        filtered_dns_shell_power, min_wavenumber=inverse_min_wavenumber,
    )
    predicted_dns = reconstruct_germano_torch \
    (
        inverse_input, alpha, lx, ly, inverse_method,
        van_cittert_iterations, van_cittert_relaxation,
    )
    predicted_dns = post_filter_torch \
    (
        predicted_dns, lx, ly, post_filter_type,
        post_cutoff_fraction, post_cutoff_order,
    )
    if vorticity_from_potential:
        predicted_dns = torch.cat \
        (
             \
            (
                predicted_dns,
                periodic_laplacian_2nd_torch(predicted_dns[:, 1:2], lx, ly),
            ), dim=1
        )
    value_error = torch.abs \
    (
        (predicted_dns[:, :2] - dns_target[:, :2]) / dns_std[:, :2]
    )
    density_loss = value_error[:, 0].mean()
    potential_loss = value_error[:, 1].mean()
    value_loss = value_error.mean()
    vorticity_loss = torch.abs \
    (
        (predicted_dns[:, 2] - dns_target[:, 2]) / vorticity_std
    ).mean()
    value_mse = value_error.square().mean()
    vorticity_mse = \
    (
        (predicted_dns[:, 2] - dns_target[:, 2]) / vorticity_std
    ).square().mean()
    total = \
    (
        value_loss + lambda_mse * value_mse
        + lambda_vorticity * (vorticity_loss + lambda_mse * vorticity_mse)
        + lambda_kl * kl
    )
    values = \
    {
        "total": total,
        "dns": value_loss,
        "dns_mse": value_mse,
        "dns_P_mse": vorticity_mse,
        "kl": kl,
        "dns_U": density_loss,
        "dns_V": potential_loss,
        "dns_P": vorticity_loss,
    }
    return total, values, predicted_filtered, predicted_dns


def save_training_sample_diagnostics \
(
    source_path,
    field_names,
    fine_n,
    coarse_n,
    alpha,
    lx,
    ly,
    dtype,
    output_dir,
):
    """Save one deterministic DNS/filter/downsampling training example."""
    dns = load_dns_snapshot \
    (
        source_path,
        field_names,
        fine_n,
        lx,
        ly,
        dtype=dtype,
    )
    coarse, filtered = filter_snapshot \
    (
        dns,
        alpha,
        lx,
        ly,
        coarse_n,
        dtype=dtype,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stages = \
    (
        ("DNS", dns),
        (f"Filtered {fine_n}", filtered),
        (f"Filtered and downsampled {coarse_n}", coarse),
    )

    fig, axes = plt.subplots \
    (
        len(stages),
        len(field_names),
        figsize=(5.2 * len(field_names), 4.5 * len(stages)),
        squeeze=False,
    )
    for row, (stage, values) in enumerate(stages):
        for channel, name in enumerate(field_names):
            ax = axes[row, channel]
            image = ax.imshow \
            (
                values[channel].T,
                origin="lower",
                extent=(0.0, lx, 0.0, ly),
                aspect="equal",
                cmap={"U": "Blues", "V": "RdBu_r", "P": "hot"}.get \
                (
                    name, "viridis"
                ),
            )
            ax.set_title(f"{stage}: {HW_FIELD_LABELS.get(name, name)}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            fig.colorbar(image, ax=ax)
    fig.suptitle(f"Training sample: {source_path}", fontsize=11)
    fig.tight_layout()
    fields_path = output_dir / "training_sample_fields.png"
    fig.savefig(fields_path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots \
    (
        1, len(field_names), figsize=(5.8 * len(field_names), 5.0),
        squeeze=False,
    )
    for channel, (ax, name) in enumerate(zip(axes[0], field_names)):
        for stage, values in stages:
            nx, ny = values[channel].shape
            x = np.arange(nx) * lx / nx
            ax.plot \
            (
                x,
                values[channel, :, ny // 2],
                linewidth=1.4,
                marker="o" if nx == coarse_n else None,
                markersize=2.5,
                label=f"{stage} (y={ly * (ny // 2) / ny:g})",
            )
        ax.set_title(f"{HW_FIELD_LABELS.get(name, name)} centerline")
        ax.set_xlabel("x")
        ax.set_ylabel(HW_FIELD_LABELS.get(name, name))
        ax.grid(True, alpha=0.25)
        ax.legend()
    fig.tight_layout()
    centerline_path = output_dir / "training_sample_centerlines.png"
    fig.savefig(centerline_path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots \
    (
        1, len(field_names), figsize=(5.8 * len(field_names), 5.0),
        squeeze=False,
    )
    for channel, (ax, name) in enumerate(zip(axes[0], field_names)):
        for stage, values in stages:
            k, power = isotropic_spectrum \
            (
                values[channel], lx, ly, smooth=True
            )
            positive = (k > 0) & (power > 0)
            ax.loglog \
            (
                k[positive],
                power[positive],
                linewidth=1.5,
                marker="o" if values.shape[-1] == coarse_n else None,
                markersize=3,
                label=stage,
            )
        coarse_nyquist = min \
        (
            np.pi / (lx / coarse_n), np.pi / (ly / coarse_n)
        )
        ax.axvline \
        (
            coarse_nyquist,
            color="0.4",
            linestyle=":",
            linewidth=1.2,
            label=f"{coarse_n}-grid Nyquist",
        )
        ax.set_title(f"{HW_FIELD_LABELS.get(name, name)} spectrum")
        ax.set_xlabel(r"wavenumber $k$")
        ax.set_ylabel("shell-summed power")
        ax.set_ylim(bottom=SPECTRUM_Y_MIN, top=SPECTRUM_Y_MAX)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
    fig.tight_layout()
    spectra_path = output_dir / "training_sample_spectra.png"
    fig.savefig(spectra_path, dpi=180)
    plt.close(fig)

    print \
    (
        "Saved first training-sample fields, centerlines, and spectra to "
        f"{output_dir} (source: {source_path})."
    )
    return fields_path, centerline_path, spectra_path


@torch.inference_mode()
def save_epoch_diagnostics \
(
    epoch,
    model,
    dataset,
    device,
    mean,
    std,
    dns_std,
    vorticity_std,
    alpha,
    lx,
    ly,
    lambda_dns,
    lambda_first_derivative,
    lambda_second_derivative,
    lambda_vorticity,
    vorticity_from_potential,
    inverse_cutoff_fraction,
    inverse_cutoff_order,
    inverse_filter_type,
    filtered_dns_shell_power,
    post_cutoff_fraction,
    post_cutoff_order,
    post_filter_type,
    inverse_method,
    van_cittert_iterations,
    van_cittert_relaxation,
    field_names,
    output_root,
    inverse_min_wavenumber=1.0,
):
    """Save model and DNS diagnostics for a fixed validation snapshot."""
    coarse, filtered_target, dns_target = dataset[0]
    coarse_batch = coarse.unsqueeze(0).to(device)
    filtered_batch = filtered_target.unsqueeze(0).to(device)
    dns_batch = dns_target.unsqueeze(0).to(device)
    model.eval()
    _, _, predicted_filtered, reconstructed_dns = calculate_training_losses \
    (
        model,
        coarse_batch,
        filtered_batch,
        dns_batch,
        mean,
        std,
        dns_std,
        vorticity_std,
        alpha,
        lx,
        ly,
        lambda_dns,
        lambda_first_derivative,
        lambda_second_derivative,
        lambda_vorticity,
        vorticity_from_potential,
        inverse_cutoff_fraction,
        inverse_cutoff_order,
        inverse_filter_type,
        filtered_dns_shell_power,
        post_cutoff_fraction,
        post_cutoff_order,
        post_filter_type,
        inverse_method,
        van_cittert_iterations,
        van_cittert_relaxation,
        inverse_min_wavenumber=inverse_min_wavenumber,
    )
    coarse_physical = (coarse_batch * std + mean)[0].cpu().numpy()
    predicted_filtered = predicted_filtered[0].cpu().numpy()
    if vorticity_from_potential:
        predicted_filtered = np.concatenate \
        (
            (
                predicted_filtered,
                periodic_laplacian_2nd_numpy \
                (
                    predicted_filtered[1:2], lx, ly
                ),
            ),
            axis=0,
        )
    reconstructed_dns = reconstructed_dns[0].cpu().numpy()
    dns_target = dns_batch[0].cpu().numpy()

    output_dir = output_root / f"epoch_{epoch:05d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_field_images \
    (
        coarse_physical,
        field_names,
        output_dir / "unet_input.png",
        "U-Net input",
    )
    save_field_images \
    (
        predicted_filtered,
        field_names,
        output_dir / "unet_output.png",
        "U-Net output",
    )
    save_field_images \
    (
        reconstructed_dns,
        field_names,
        output_dir / "reconstructed_dns.png",
        "Reconstructed DNS",
    )
    save_field_error_images \
    (
        reconstructed_dns,
        dns_target,
        field_names,
        output_dir / "dns_error.png",
    )
    save_field_centerline_comparison \
    (
        reconstructed_dns,
        dns_target,
        field_names,
        output_dir / "dns_centerlines.png",
    )
    save_field_spectrum_comparison \
    (
        reconstructed_dns,
        dns_target,
        field_names,
        lx,
        ly,
        output_dir / "dns_spectra.png",
        post_cutoff_fraction,
    )
    print(f"Saved epoch {epoch} validation images to {output_dir}.")


class Tee:
    """Write a text stream to both the terminal and a log file."""

    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, text):
        self.terminal.write(text)
        self.log_file.write(text)
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self):
        return self.terminal.isatty()


def split_files(dataset, validation_fraction, test_fraction, split_mode, seed):
    td, vd, sd = dataset / "train", dataset / "validation", dataset / "test"
    if td.is_dir() and vd.is_dir() and sd.is_dir():
        return \
        (
            recursive_npz_files(td),
            recursive_npz_files(vd),
            recursive_npz_files(sd),
        )
    files = recursive_npz_files(dataset)
    if len(files) < 3:
        raise ValueError("At least three NPZ snapshots are required.")
    if split_mode == "random":
        random.Random(seed).shuffle(files)
    nv = max(1, round(validation_fraction * len(files)))
    nt = max(1, round(test_fraction * len(files)))
    ntr = len(files) - nv - nt
    if ntr < 1:
        raise ValueError("The split leaves no training snapshots.")
    return files[:ntr], files[ntr : ntr + nv], files[ntr + nv :]


def run_epoch \
(
    model,
    loader,
    device,
    mean,
    std,
    dns_std,
    vorticity_std,
    alpha,
    lx,
    ly,
    lambda_dns,
    lambda_first_derivative,
    lambda_second_derivative,
    lambda_vorticity,
    vorticity_from_potential,
    inverse_cutoff_fraction,
    inverse_cutoff_order,
    inverse_filter_type,
    filtered_dns_shell_power,
    post_cutoff_fraction,
    post_cutoff_order,
    post_filter_type,
    inverse_method,
    van_cittert_iterations,
    van_cittert_relaxation,
    optimizer=None,
    lambda_kl=0.0,
    inverse_min_wavenumber=1.0,
    lambda_mse=0.1,
):
    training = optimizer is not None
    model.train(training)
    loss_names = \
    [
        "total",
        "dns",
        "dns_U",
        "dns_V",
        "dns_P",
        "dns_mse",
        "dns_P_mse",
        "kl",
    ]
    totals = \
    {
        name: 0.0
        for name in loss_names
        + ["filtered_relative_l2", "dns_relative_l2"]
    }
    count = 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for coarse, ft, dns in loader:
            coarse, ft, dns = \
            (
                coarse.to(device, non_blocking=True),
                ft.to(device, non_blocking=True),
                dns.to(device, non_blocking=True),
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
            loss, vals, pf, pd = calculate_training_losses \
            (
                model,
                coarse,
                ft,
                dns,
                mean,
                std,
                dns_std,
                vorticity_std,
                alpha,
                lx,
                ly,
                lambda_dns,
                lambda_first_derivative,
                lambda_second_derivative,
                lambda_vorticity,
                vorticity_from_potential,
                inverse_cutoff_fraction,
                inverse_cutoff_order,
                inverse_filter_type,
                filtered_dns_shell_power,
                post_cutoff_fraction,
                post_cutoff_order,
                post_filter_type,
                inverse_method,
                van_cittert_iterations,
                van_cittert_relaxation,
                lambda_kl=lambda_kl,
                lambda_mse=lambda_mse,
                inverse_min_wavenumber=inverse_min_wavenumber,
            )
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            b = coarse.shape[0]
            count += b
            for k in loss_names:
                totals[k] += vals[k].detach().item() * b
            totals["filtered_relative_l2"] += \
            (
                relative_l2 \
                (
                    (ft * std + mean)[:, :pf.shape[1]], pf
                ).detach().item() * b
            )
            totals["dns_relative_l2"] += \
            (
                relative_l2(dns[:, :2], pd[:, :2]).detach().item() * b
            )

    return {k: v / count for k, v in totals.items()}


def configure_tensorboard(writer, normalize):
    writer.add_custom_scalars \
    (
        {
            "Losses": \
            {
                name.title(): \
                [
                    "Multiline",
                    [f"metrics/train/{name}", f"metrics/validation/{name}"],
                ]
                for name in \
                (
                    "total",
                    "dns",
                    "dns_mse",
                    "dns_P_mse",
                )
            },
            "Relative L2": \
            {
                name.replace("_", " ").title(): \
                [
                    "Multiline",
                    [f"metrics/train/{name}", f"metrics/validation/{name}"],
                ]
                for name in \
                (
                    "filtered_relative_l2",
                    "dns_relative_l2",
                )
            },
            (
                "Per-field DNS loss (vorticity normalized)"
                if normalize
                else "Per-field DNS loss (U/V absolute, vorticity normalized)"
            ): \
            {
                name: \
                [
                    "Multiline",
                    [
                        f"metrics/train/dns_{name}",
                        f"metrics/validation/dns_{name}",
                    ],
                ]
                for name in HW_FIELD_NAMES
            },
        }
    )


def log_epoch_metrics(writer, epoch, train_metrics, validation_metrics, lr):
    for name, value in train_metrics.items():
        writer.add_scalar(f"metrics/train/{name}", value, epoch)
    for name, value in validation_metrics.items():
        writer.add_scalar(f"metrics/validation/{name}", value, epoch)
    writer.add_scalar("optimizer/learning_rate", lr, epoch)
    writer.flush()


def serialise_parameters(args, **derived):
    parameters = {}
    for name, value in vars(args).items():
        if isinstance(value, Path):
            value = str(value)
        parameters[name] = value
    parameters.update(derived)
    return parameters


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument \
    (
        "--output-dir", type=Path, default=Path("germano_training_output")
    )
    p.add_argument("--fields", nargs=3, default=list(HW_FIELD_NAMES))
    p.add_argument \
    (
        "--fine-n",
        type=int,
        default=256,
        help="DNS grid resolution per direction (default: 256)",
    )
    p.add_argument \
    (
        "--coarse-n",
        type=int,
        default=64,
        help= \
        (
            "LES grid resolution per direction; must divide --fine-n "
            "(default: 64)"
        ),
    )
    p.add_argument("--lx", type=float, default=2 * np.pi)
    p.add_argument("--ly", type=float, default=2 * np.pi)
    p.add_argument \
    (
        "--alpha",
        type=float,
        help= \
        (
            "Physical Germano filter width "
            "(default: one fine-grid spacing, Lx/fine-n)"
        ),
    )
    p.add_argument \
    (
        "--split-mode", choices=["sequential", "random"], default="random"
    )
    p.add_argument("--validation-fraction", type=float, default=0.1)
    p.add_argument("--test-fraction", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=1e-4)
    p.add_argument \
    (
        "--learning-rate-policy",
        choices=("constant", "step"),
        default="constant",
        help="Learning-rate policy (default: constant/no changes)",
    )
    p.add_argument \
    (
        "--learning-rate-step",
        type=int,
        default=25,
        help="Epochs between stepwise learning-rate reductions (default: 25)",
    )
    p.add_argument \
    (
        "--learning-rate-gamma",
        type=float,
        default=0.5,
        help="Learning-rate multiplier at each step (default: 0.5)",
    )
    p.add_argument("--base-channels", type=int, default=24)
    p.add_argument \
    (
        "--variational", action=argparse.BooleanOptionalAction, default=False,
        help="Use a Gaussian variational bottleneck (default: disabled)"
    )
    p.add_argument \
    (
        "--lambda-mse", type=float, default=0.1,
        help="Weight of squared normalized DNS errors added to L1; vorticity "
        "also uses --lambda-vorticity (default: 0.1; 0 restores L1 only)",
    )
    p.add_argument \
    (
        "--lambda-kl", type=float, default=0.0,
        help="Weight of mean latent KL divergence (default: 0, disabled)"
    )
    p.add_argument \
    (
        "--precision",
        choices=("float32", "float64"),
        default="float32",
        help="Floating-point precision for data, U-Net, losses, and physics",
    )
    p.add_argument \
    (
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help= \
        (
            "Normalize LES inputs and filtered targets and scale reconstruction loss "
            "by its standard deviation (default: enabled; use "
            "--no-normalize for physical values throughout)"
        ),
    )
    p.add_argument \
    (
        "--lambda-dns", type=float, default=1.0,
        help="Deprecated and ignored; the DNS n/phi loss has unit weight",
    )
    p.add_argument \
    (
        "--lambda-first-derivative", type=float, default=1.0e-2,
        help="Deprecated and ignored; derivative losses are disabled",
    )
    p.add_argument \
    (
        "--lambda-second-derivative", type=float, default=1.0e-4,
        help="Deprecated and ignored; derivative losses are disabled",
    )
    p.add_argument \
    (
        "--lambda-vorticity", type=float, default=0.0,
        help= \
        (
            "Weight of the DNS vorticity L1 error normalized by the training "
            "vorticity standard deviation (default: 0, disabled)"
        ),
    )
    p.add_argument \
    (
        "--vorticity-from-potential",
        action=argparse.BooleanOptionalAction,
        default=True,
        help= \
        (
            "Derive DNS vorticity as the periodic second-order Laplacian "
            "of reconstructed potential and predict only U,V (default: "
            "enabled; use --no-vorticity-from-potential for a learned P)"
        ),
    )
    p.add_argument \
    (
        "--random-periodic-shift",
        action=argparse.BooleanOptionalAction,
        default=True,
        help= \
        (
            "Apply synchronized random periodic translations to training "
            "U, V, and P fields (default: enabled)"
        ),
    )
    p.add_argument \
    (
        "--periodic-x",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Declare the x direction periodic (default: enabled)",
    )
    p.add_argument \
    (
        "--periodic-y",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Declare the y direction periodic (default: enabled)",
    )
    p.add_argument \
    (
        "--inverse-cutoff-fraction",
        type=float,
        default=0.0,
        help= \
        (
            "Smooth inverse cutoff as a fraction of Nyquist "
            "(default: disabled; 0 disables it)"
        ),
    )
    p.add_argument("--inverse-cutoff-order", type=float, default=8.0)
    p.add_argument \
    (
        "--inverse-min-wavenumber", type=float, default=1.0,
        help="Physical k threshold for filtered_dns_spectrum: modes k <= this "
        "pass unchanged (default: 1.0; 0 protects only the mean)",
    )
    p.add_argument \
    (
        "--inverse-filter",
        choices=INVERSE_FILTER_TYPES,
        default="cutoff",
        help= \
        (
            "Regularization applied to the U-Net output before inversion: "
            "the existing analytic cutoff, or a suppressive shell filter "
            "matched to the mean filtered-DNS training spectrum"
        ),
    )
    p.add_argument \
    (
        "--inverse-method",
        choices=INVERSE_METHODS,
        default="exact",
        help="DNS reconstruction operator used in training (default: exact)",
    )
    p.add_argument \
    (
        "--van-cittert-iterations",
        type=int,
        default=1,
        help="Number of van Cittert correction steps (default: 1)",
    )
    p.add_argument \
    (
        "--van-cittert-relaxation",
        type=float,
        default=1.0,
        help="Van Cittert relaxation in (0, 1] (default: 1)",
    )
    p.add_argument \
    (
        "--post-filter",
        choices=POST_FILTER_TYPES,
        default="high_order",
        help= \
        (
            "Post-inversion filter: none, current high-order exponential, "
            "or Gaussian "
            "(default: high_order)"
        ),
    )
    p.add_argument \
    (
        "--post-cutoff-fraction",
        type=float,
        default=0.0,
        help= \
        (
            "Post-inversion smooth cutoff as a fraction of Nyquist; "
            "0 disables it (default: disabled)"
        ),
    )
    p.add_argument \
    (
        "--post-cutoff-order",
        type=float,
        default=8.0,
        help="Post-inversion smooth cutoff order (default: 8)",
    )
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument \
    (
        "--save-every",
        type=int,
        default=25,
        help="Save an epoch-numbered checkpoint every N epochs; 0 disables it",
    )
    p.add_argument \
    (
        "--diagnostic-every",
        type=int,
        default=5,
        help="Save training diagnostic images every N epochs; 0 disables them",
    )
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--resume", type=Path)
    p.add_argument \
    (
        "--tensorboard-dir",
        type=Path,
        help="TensorBoard log directory (default: OUTPUT_DIR/tensorboard)",
    )
    p.add_argument \
    (
        "--log-file",
        type=Path,
        help="Console log file (default: OUTPUT_DIR/training.log)",
    )
    return p.parse_args()


def run_training(a):
    set_seed(a.seed)
    torch_dtype = getattr(torch, a.precision)
    numpy_dtype = getattr(np, a.precision)
    if tuple(a.fields) != HW_FIELD_NAMES:
        raise ValueError(f"Hasegawa-Wakatani fields must be {HW_FIELD_NAMES}.")
    scale_factor = validate_grid_pair(a.fine_n, a.coarse_n)
    if a.alpha is not None and (not np.isfinite(a.alpha) or a.alpha < 0):
        raise ValueError("alpha must be finite and non-negative.")
    if not 0 <= a.inverse_cutoff_fraction <= 1:
        raise ValueError("inverse-cutoff-fraction must be in [0, 1].")
    if a.inverse_cutoff_order <= 0:
        raise ValueError("inverse-cutoff-order must be positive.")
    if a.van_cittert_iterations < 0:
        raise ValueError("van-cittert-iterations must be non-negative.")
    if not 0 < a.van_cittert_relaxation <= 1:
        raise ValueError("van-cittert-relaxation must be in (0, 1].")
    if not 0 <= a.post_cutoff_fraction <= 1:
        raise ValueError("post-cutoff-fraction must be in [0, 1].")
    if a.post_cutoff_order <= 0:
        raise ValueError("post-cutoff-order must be positive.")
    if a.learning_rate_step <= 0:
        raise ValueError("learning-rate-step must be positive.")
    if not 0 < a.learning_rate_gamma <= 1:
        raise ValueError("learning-rate-gamma must be in (0, 1].")
    if a.lambda_first_derivative < 0:
        raise ValueError("lambda-first-derivative must be non-negative.")
    if a.lambda_second_derivative < 0:
        raise ValueError("lambda-second-derivative must be non-negative.")
    if not np.isfinite(a.lambda_mse) or a.lambda_mse < 0:
        raise ValueError("--lambda-mse must be finite and nonnegative")
    if not np.isfinite(a.lambda_kl) or a.lambda_kl < 0:
        raise ValueError("--lambda-kl must be finite and nonnegative")
    if not np.isfinite(a.lambda_vorticity) or a.lambda_vorticity < 0:
        raise ValueError("lambda-vorticity must be finite and non-negative.")
    all_directions_periodic = a.periodic_x and a.periodic_y
    use_random_periodic_shift = \
    (
        a.random_periodic_shift and all_directions_periodic
    )
    if a.random_periodic_shift and not all_directions_periodic:
        nonperiodic = \
        [
            direction
            for direction, periodic in \
            (
                ("x", a.periodic_x),
                ("y", a.periodic_y),
            )
            if not periodic
        ]
        print \
        (
            "Random periodic shift disabled because not all directions are "
            f"periodic (nonperiodic: {', '.join(nonperiodic)})."
        )
    if not np.isfinite(a.inverse_min_wavenumber) or a.inverse_min_wavenumber < 0:
        raise ValueError("inverse-min-wavenumber must be finite and nonnegative")
    inverse_cutoff_fraction = a.inverse_cutoff_fraction or None
    post_cutoff_fraction = a.post_cutoff_fraction or None
    if a.post_filter in ("high_order", "gaussian"):
        if post_cutoff_fraction is None:
            print(f"Post filter {a.post_filter} disabled: cutoff fraction is 0.")
    elif a.post_filter == "none":
        post_cutoff_fraction = None
    a.output_dir.mkdir(parents=True, exist_ok=True)
    ckdir = a.output_dir / "checkpoints"
    ckdir.mkdir(exist_ok=True)
    parameters_path = a.output_dir / "training_parameters.json"
    tr, va, te = split_files \
    (
        a.dataset, a.validation_fraction, a.test_fraction, a.split_mode, a.seed
    )
    delta = a.lx / a.fine_n
    alpha = a.alpha if a.alpha is not None else delta
    model_output_channels = 2 if a.vorticity_from_potential else 3
    filtered_dns_shell_power_np = None
    if a.inverse_filter == "filtered_dns_spectrum":
        print("Calculating mean filtered-DNS spectrum from training snapshots.")
        filtered_dns_shell_power_np = calculate_filtered_dns_shell_power \
        (
            tr, a.fields, a.fine_n, a.coarse_n, alpha, a.lx, a.ly,
            model_output_channels, dtype=numpy_dtype,
        )
    if a.normalize:
        mean_np, std_np, dns_std_np = calculate_training_statistics \
        (
            tr,
            a.fields,
            a.fine_n,
            a.coarse_n,
            alpha,
            a.lx,
            a.ly,
            dtype=numpy_dtype,
        )
    else:
        statistics_shape = (1, len(a.fields), 1, 1)
        mean_np = np.zeros(statistics_shape, dtype=numpy_dtype)
        std_np = np.ones(statistics_shape, dtype=numpy_dtype)
        dns_std_np = np.ones(statistics_shape, dtype=numpy_dtype)
        print \
        (
            "Normalization disabled: LES inputs, filtered targets, model "
            "outputs, and reconstruction value losses use physical values."
        )
    if a.normalize:
        vorticity_std_np = dns_std_np[:, 2:3]
    elif a.lambda_vorticity > 0:
        print("Calculating vorticity normalization for vorticity loss.")
        _, _, physical_dns_std_np = calculate_training_statistics \
        (
            tr, a.fields, a.fine_n, a.coarse_n, alpha, a.lx, a.ly,
            dtype=numpy_dtype,
        )
        vorticity_std_np = physical_dns_std_np[:, 2:3]
    else:
        vorticity_std_np = np.ones((1, 1, 1, 1), dtype=numpy_dtype)

    def ds(files, random_periodic_shift=False):
        return GermanoSnapshotDataset \
        (
            files,
            a.fields,
            a.fine_n,
            a.coarse_n,
            alpha,
            a.lx,
            a.ly,
            mean_np,
            std_np,
            random_periodic_shift,
            (a.periodic_x, a.periodic_y),
        )

    train_ds = ds(tr, use_random_periodic_shift)
    val_ds, test_ds = ds(va), ds(te)
    kw = dict \
    (
        batch_size=a.batch_size,
        num_workers=a.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=a.num_workers > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, **kw)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print("Precision:", a.precision)
    print \
    (
        "Training objective: mean absolute DNS-space error in electron "
        "density and electrostatic potential plus "
        f"{a.lambda_mse:g} times their normalized MSE, plus "
        f"{a.lambda_vorticity:g} times (normalized vorticity L1 "
        f"+ {a.lambda_mse:g} times normalized vorticity MSE) "
        f"plus {a.lambda_kl if a.variational else 0:g} times mean latent KL."
    )
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.conv.fp32_precision = "ieee"
    model = LegacyResidualUNetSR \
    (
        len(a.fields),
        a.base_channels,
        scale_factor,
        model_output_channels,
        variational=a.variational,
    ).to(device=device, dtype=torch_dtype)
    opt = torch.optim.AdamW \
    (
        model.parameters(), lr=a.learning_rate, weight_decay=1e-6
    )
    if a.learning_rate_policy == "step":
        scheduler = torch.optim.lr_scheduler.StepLR \
        (
            opt,
            step_size=a.learning_rate_step,
            gamma=a.learning_rate_gamma,
        )
    else:
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1.0)
    mean = torch.from_numpy(mean_np).to(device)
    std = torch.from_numpy(std_np).to(device)
    dns_std = torch.from_numpy(dns_std_np).to(device)
    vorticity_std = torch.as_tensor \
    (
        vorticity_std_np.item(), device=device, dtype=torch_dtype
    )
    filtered_dns_shell_power = \
    (
        None if filtered_dns_shell_power_np is None
        else torch.from_numpy(filtered_dns_shell_power_np).to(device)
    )
    start = 1
    best = float("inf")
    if a.resume:
        c = load_checkpoint(a.resume, device)
        validate_hw_checkpoint(c)
        if checkpoint_model_class(c) is not LegacyResidualUNetSR:
            raise ValueError \
            (
                "Cannot resume a different architecture as a LegacyResidualUNetSR. "
                "Start fresh training without --resume."
            )
        if c.get("variational", False) != a.variational:
            raise ValueError("Resume variational setting does not match the checkpoint.")
        checkpoint_derives_vorticity = \
        (
            c["vorticity_mode"] == "laplacian_of_potential"
        )
        if checkpoint_derives_vorticity != a.vorticity_from_potential:
            raise ValueError \
            (
                "--vorticity-from-potential setting does not match the "
                "resumed checkpoint."
            )
        checkpoint_precision = c.get("precision", "float32")
        if checkpoint_precision != a.precision:
            raise ValueError \
            (
                f"--precision {a.precision} does not match resumed "
                f"checkpoint precision {checkpoint_precision}."
            )
        checkpoint_normalizes = c.get \
        (
            "normalize",
            c.get("training_parameters", {}).get("normalize", True),
        )
        if checkpoint_normalizes != a.normalize:
            raise ValueError \
            (
                f"--{'normalize' if a.normalize else 'no-normalize'} does "
                "not match the resumed checkpoint."
            )
        model.load_state_dict(c["model"])
        restore_checkpoint_model_settings(model, c)
        opt.load_state_dict(c["optimizer"])
        if "scheduler" in c:
            scheduler_state = c["scheduler"]
            checkpoint_uses_step = \
            (
                "step_size" in scheduler_state and "gamma" in scheduler_state
            )
            requested_uses_step = a.learning_rate_policy == "step"
            if checkpoint_uses_step != requested_uses_step:
                checkpoint_policy = "step" if checkpoint_uses_step else "constant"
                raise ValueError \
                (
                    "--learning-rate-policy does not match the resumed "
                    f"checkpoint: checkpoint uses {checkpoint_policy!r}, but "
                    f"the command requested {a.learning_rate_policy!r}."
                )
            scheduler.load_state_dict(c["scheduler"])
        else:
            completed_epoch = int(c["epoch"])
            resumed_lr = a.learning_rate
            if a.learning_rate_policy == "step":
                resumed_lr *= a.learning_rate_gamma ** \
                (
                    completed_epoch // a.learning_rate_step
                )
            for group in opt.param_groups:
                group["lr"] = resumed_lr
            scheduler.last_epoch = completed_epoch
            scheduler._last_lr = [resumed_lr for _ in opt.param_groups]
        start = int(c["epoch"]) + 1
        best = float(c["validation_loss"])
        if c.get("training_objective") != \
        (
            "dns_l1_mse_density_potential_plus_normalized_vorticity_plus_kl"
        ):
            print("Training objective changed; resetting best validation score.")
            best = float("inf")
        if c.get("inverse_min_wavenumber", 1.0) != a.inverse_min_wavenumber:
            best = float("inf")
        if c.get("lambda_mse", 0.0) != a.lambda_mse:
            best = float("inf")
        if c.get("lambda_kl", 0.0) != a.lambda_kl:
            best = float("inf")
        checkpoint_lambda_vorticity = float \
        (
            c.get \
            (
                "lambda_vorticity",
                c.get("training_parameters", {}).get("lambda_vorticity", 0.0),
            )
        )
        if checkpoint_lambda_vorticity != a.lambda_vorticity:
            print \
            (
                "Vorticity loss weight changed from "
                f"{checkpoint_lambda_vorticity:g} to {a.lambda_vorticity:g}; "
                "resetting best validation score for the new objective."
            )
            best = float("inf")
        print("Resumed from epoch", start - 1)
    tensorboard_dir = a.tensorboard_dir or a.output_dir / "tensorboard"
    writer = SummaryWriter(log_dir=tensorboard_dir, purge_step=start)
    configure_tensorboard(writer, a.normalize)
    print("TensorBoard logs:", tensorboard_dir)
    parameters = serialise_parameters \
    (
        a,
        alpha=alpha,
        grid_spacing=delta,
        scale_factor=scale_factor,
        optimizer="AdamW",
        optimizer_learning_rate=opt.param_groups[0]["lr"],
        learning_rate_policy=a.learning_rate_policy,
        learning_rate_step=a.learning_rate_step,
        learning_rate_gamma=a.learning_rate_gamma,
        optimizer_weight_decay=opt.param_groups[0]["weight_decay"],
        optimizer_betas=opt.param_groups[0]["betas"],
        optimizer_epsilon=opt.param_groups[0]["eps"],
        optimizer_amsgrad=opt.param_groups[0]["amsgrad"],
        gradient_clip_norm=1.0,
        model="LegacyResidualUNetSR",
        variational=a.variational,
        lambda_kl=a.lambda_kl,
        lambda_mse=a.lambda_mse,
        model_input_channels=len(a.fields),
        model_output_channels=model_output_channels,
        trainable_model_parameters=sum \
        (
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ),
        differential_filter="1 / (1 + alpha^2 k^2)",
        downsampling="centered Fourier-mode truncation",
        vorticity_operator= \
        (
            "second_order_periodic_laplacian"
            if a.vorticity_from_potential
            else "learned_independent_channel"
        ),
        vorticity_mode= \
        (
            "laplacian_of_potential"
            if a.vorticity_from_potential
            else "learned_channel"
        ),
        dns_vorticity_mode= \
        (
            "derived_from_potential"
            if a.vorticity_from_potential
            else "independent_channel"
        ),
        convolution_padding="circular",
        interpolation_boundary="periodic",
        periodic_directions={"x": a.periodic_x, "y": a.periodic_y},
        all_directions_periodic=all_directions_periodic,
        random_periodic_shift_requested=a.random_periodic_shift,
        random_periodic_shift_enabled=use_random_periodic_shift,
        training_augmentation= \
        (
            "synchronized_random_periodic_shift"
            if use_random_periodic_shift
            else "none"
        ),
        inverse_method=a.inverse_method,
        lambda_dns=1.0,
        lambda_first_derivative=0.0,
        lambda_second_derivative=0.0,
        lambda_vorticity=a.lambda_vorticity,
        training_objective= \
        (
            "dns_l1_mse_density_potential_plus_normalized_vorticity_plus_kl"
        ),
        loss_channels=list(a.fields),
        effective_loss_weights= \
        {
            "dns_density_potential": 1.0,
            "dns_density_potential_mse": a.lambda_mse,
            "vorticity_mse": a.lambda_vorticity * a.lambda_mse,
            "kl": a.lambda_kl if a.variational else 0.0,
            "dns": 1.0,
            "vorticity": a.lambda_vorticity,
            "first_derivative": 0.0,
            "second_derivative": 0.0,
        },
        van_cittert_iterations=a.van_cittert_iterations,
        van_cittert_relaxation=a.van_cittert_relaxation,
        inverse_regularization= \
        (
            "filtered_dns_spectrum"
            if a.inverse_filter == "filtered_dns_spectrum"
            else "truncated_relaxed_van_cittert"
            if a.inverse_method == "van_cittert"
            else "smooth_exponential_cutoff"
        ),
        inverse_cutoff_fraction=inverse_cutoff_fraction,
        inverse_filter=a.inverse_filter,
        filtered_dns_spectrum_shells= \
        (
            None if filtered_dns_shell_power_np is None
            else filtered_dns_shell_power_np.shape[1]
        ),
        inverse_cutoff_wavenumber= \
        (
            inverse_cutoff_fraction * np.pi / delta
            if inverse_cutoff_fraction is not None
            else None
        ),
        post_filter=a.post_filter,
        post_cutoff_fraction=post_cutoff_fraction,
        post_cutoff_order=a.post_cutoff_order,
        physics_version=1,
        train_snapshots=len(tr),
        validation_snapshots=len(va),
        test_snapshots=len(te),
        normalization_mean=mean_np.reshape(-1).tolist(),
        normalization_std=std_np.reshape(-1).tolist(),
        dns_normalization_std=dns_std_np.reshape(-1).tolist(),
        vorticity_normalization_std=float(vorticity_std_np.item()),
        normalization_enabled=a.normalize,
        device=str(device),
        gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        tensorboard_dir=str(tensorboard_dir),
    )
    parameters_path.write_text \
    (
        json.dumps(parameters, indent=2, sort_keys=True)
    )
    def make_checkpoint(epoch, validation_loss):
        return \
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "architecture": "LegacyResidualUNetSR",
            "global_input_skip": model.global_input_skip,
            "variational": a.variational,
            "lambda_kl": a.lambda_kl,
            "lambda_mse": a.lambda_mse,
            "optimizer": opt.state_dict(),
            "scheduler": scheduler.state_dict(),
            "validation_loss": validation_loss,
            "alpha": alpha,
            "delta": delta,
            "mean": mean_np,
            "std": std_np,
            "dns_std": dns_std_np,
            "vorticity_std": vorticity_std_np,
            "field_names": list(a.fields),
            "physics_version": 1,
            "vorticity_operator": parameters["vorticity_operator"],
            "vorticity_mode": parameters["vorticity_mode"],
            "dns_vorticity_mode": parameters["dns_vorticity_mode"],
            "model_output_channels": model_output_channels,
            "training_parameters": parameters,
            "fine_n": a.fine_n,
            "coarse_n": a.coarse_n,
            "lx": a.lx,
            "ly": a.ly,
            "base_channels": a.base_channels,
            "lambda_dns": 1.0,
            "lambda_first_derivative": 0.0,
            "lambda_second_derivative": 0.0,
            "lambda_vorticity": a.lambda_vorticity,
            "training_objective": parameters["training_objective"],
            "loss_channels": parameters["loss_channels"],
            "effective_loss_weights": parameters["effective_loss_weights"],
            "learning_rate_step": a.learning_rate_step,
            "learning_rate_gamma": a.learning_rate_gamma,
            "learning_rate_policy": a.learning_rate_policy,
            "periodic_x": a.periodic_x,
            "periodic_y": a.periodic_y,
            "all_directions_periodic": all_directions_periodic,
            "random_periodic_shift_requested": a.random_periodic_shift,
            "random_periodic_shift": use_random_periodic_shift,
            "inverse_cutoff_fraction": inverse_cutoff_fraction,
            "inverse_min_wavenumber": a.inverse_min_wavenumber,
            "inverse_cutoff_order": a.inverse_cutoff_order,
            "inverse_filter": a.inverse_filter,
            "filtered_dns_shell_power": filtered_dns_shell_power_np,
            "inverse_method": a.inverse_method,
            "van_cittert_iterations": a.van_cittert_iterations,
            "van_cittert_relaxation": a.van_cittert_relaxation,
            "post_cutoff_fraction": post_cutoff_fraction,
            "post_cutoff_order": a.post_cutoff_order,
            "post_filter": a.post_filter,
            "precision": a.precision,
            "normalize": a.normalize,
        }

    if start == 1:
        torch.save(make_checkpoint(0, float("inf")), ckdir / "epoch_00000.pt")
        save_epoch_diagnostics \
        (
            0,
            model,
            val_ds,
            device,
            mean,
            std,
            dns_std,
            vorticity_std,
            alpha,
            a.lx,
            a.ly,
            a.lambda_dns,
            a.lambda_first_derivative,
            a.lambda_second_derivative,
            a.lambda_vorticity,
            a.vorticity_from_potential,
            inverse_cutoff_fraction,
            a.inverse_cutoff_order,
            a.inverse_filter,
            filtered_dns_shell_power,
            post_cutoff_fraction,
            a.post_cutoff_order,
            a.post_filter,
            a.inverse_method,
            a.van_cittert_iterations,
            a.van_cittert_relaxation,
            a.fields,
            a.output_dir / "training_diagnostics",
            inverse_min_wavenumber=a.inverse_min_wavenumber,
        )
    history = []
    for epoch in range(start, a.epochs + 1):
        tm = run_epoch \
        (
            model,
            train_loader,
            device,
            mean,
            std,
            dns_std,
            vorticity_std,
            alpha,
            a.lx,
            a.ly,
            a.lambda_dns,
            a.lambda_first_derivative,
            a.lambda_second_derivative,
            a.lambda_vorticity,
            a.vorticity_from_potential,
            inverse_cutoff_fraction,
            a.inverse_cutoff_order,
            a.inverse_filter,
            filtered_dns_shell_power,
            post_cutoff_fraction,
            a.post_cutoff_order,
            a.post_filter,
            a.inverse_method,
            a.van_cittert_iterations,
            a.van_cittert_relaxation,
            opt,
            lambda_kl=a.lambda_kl,
            lambda_mse=a.lambda_mse,
            inverse_min_wavenumber=a.inverse_min_wavenumber,
        )
        vm = run_epoch \
        (
            model,
            val_loader,
            device,
            mean,
            std,
            dns_std,
            vorticity_std,
            alpha,
            a.lx,
            a.ly,
            a.lambda_dns,
            a.lambda_first_derivative,
            a.lambda_second_derivative,
            a.lambda_vorticity,
            a.vorticity_from_potential,
            inverse_cutoff_fraction,
            a.inverse_cutoff_order,
            a.inverse_filter,
            filtered_dns_shell_power,
            post_cutoff_fraction,
            a.post_cutoff_order,
            a.post_filter,
            a.inverse_method,
            a.van_cittert_iterations,
            a.van_cittert_relaxation,
            lambda_kl=a.lambda_kl,
            lambda_mse=a.lambda_mse,
            inverse_min_wavenumber=a.inverse_min_wavenumber,
        )
        row = \
        {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in tm.items()},
            **{f"validation_{k}": v for k, v in vm.items()},
        }
        history.append(row)
        log_epoch_metrics \
        (
            writer,
            epoch,
            tm,
            vm,
            opt.param_groups[0]["lr"],
        )
        scheduler.step()
        c = make_checkpoint(epoch, vm["total"])
        torch.save(c, ckdir / "latest.pt")
        if vm["total"] < best:
            best = vm["total"]
            torch.save(c, ckdir / "best.pt")
        if a.save_every > 0 and epoch % a.save_every == 0:
            torch.save(c, ckdir / f"epoch_{epoch:05d}.pt")
        if a.diagnostic_every > 0 and epoch % a.diagnostic_every == 0:
            save_epoch_diagnostics \
            (
                epoch,
                model,
                val_ds,
                device,
                mean,
                std,
                dns_std,
                vorticity_std,
                alpha,
                a.lx,
                a.ly,
                a.lambda_dns,
                a.lambda_first_derivative,
                a.lambda_second_derivative,
                a.lambda_vorticity,
                a.vorticity_from_potential,
                inverse_cutoff_fraction,
                a.inverse_cutoff_order,
                a.inverse_filter,
                filtered_dns_shell_power,
                post_cutoff_fraction,
                a.post_cutoff_order,
                a.post_filter,
                a.inverse_method,
                a.van_cittert_iterations,
                a.van_cittert_relaxation,
                a.fields,
                a.output_dir / "training_diagnostics",
                inverse_min_wavenumber=a.inverse_min_wavenumber,
            )
        with (a.output_dir / "training_history.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0]))
            w.writeheader()
            w.writerows(history)
        print \
        (
            f"Epoch {epoch:05d} | train={tm['total']:.6e} | "
            f"validation={vm['total']:.6e} | "
            f"DNS relL2={vm['dns_relative_l2']:.6e}"
        )
    c = load_checkpoint(ckdir / "best.pt", device)
    model.load_state_dict(c["model"])
    test = run_epoch \
    (
        model,
        test_loader,
        device,
        mean,
        std,
        dns_std,
        vorticity_std,
        alpha,
        a.lx,
        a.ly,
        a.lambda_dns,
        a.lambda_first_derivative,
        a.lambda_second_derivative,
        a.lambda_vorticity,
        a.vorticity_from_potential,
        inverse_cutoff_fraction,
        a.inverse_cutoff_order,
        a.inverse_filter,
        filtered_dns_shell_power,
        post_cutoff_fraction,
        a.post_cutoff_order,
        a.post_filter,
        a.inverse_method,
        a.van_cittert_iterations,
        a.van_cittert_relaxation,
        lambda_kl=a.lambda_kl,
        lambda_mse=a.lambda_mse,
        inverse_min_wavenumber=a.inverse_min_wavenumber,
    )
    for name, value in test.items():
        writer.add_scalar(f"metrics/test/{name}", value, int(c["epoch"]))
    writer.flush()
    writer.close()
    (a.output_dir / "metadata.json").write_text \
    (
        json.dumps \
        (
            {
                "status": "completed",
                "best_epoch": int(c["epoch"]),
                "test_metrics": test,
                "tensorboard_dir": str(tensorboard_dir),
                "training_parameters_file": str(parameters_path),
            },
            indent=2,
        )
    )

    print("Test metrics:", test)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_file = args.log_file or args.output_dir / "training.log"
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    with args.log_file.open("a", buffering=1) as log_file:
        stdout = Tee(sys.stdout, log_file)
        stderr = Tee(sys.stderr, log_file)
        with redirect_stdout(stdout), redirect_stderr(stderr):
            print(f"Training console log: {args.log_file}")
            try:
                run_training(args)
            except Exception:
                traceback.print_exc()
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
