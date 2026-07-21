#!/usr/bin/env python3
"""
Train the existing Germano-filter super-resolution U-Net on a directory
of DNS snapshots.

Dataset layouts
===============

Layout A: predefined split directories

dataset/
    train/
        snapshot_000001.npz
        snapshot_000002.npz
        ...
    validation/
        snapshot_000101.npz
        ...
    test/
        snapshot_000201.npz
        ...

Layout B: one directory, split automatically

dataset/
    snapshot_000001.npz
    snapshot_000002.npz
    ...

The directory is scanned recursively. Each NPZ snapshot must contain three
2-D arrays, by default:

    U : [256, 256]
    V : [256, 256]
    P : [256, 256]

The existing model and losses are retained:

    DNS 256
      -> Germano filter
      -> filtered 256
      -> spectral restriction to 32
      -> residual U-Net
      -> reconstructed filtered 256
      -> inverse Germano filter
      -> reconstructed DNS 256

Example
=======

python germano_unet_dataset.py \
    --dataset /path/to/dataset \
    --epochs 300 \
    --batch-size 4 \
    --alpha-factor 1.0 \
    --coarse-n 32

For an automatic temporal split, keep the snapshots sorted by filename and use:

python germano_unet_dataset.py \
    --dataset /path/to/all_snapshots \
    --split-mode sequential

For a random split:

python germano_unet_dataset.py \
    --dataset /path/to/all_snapshots \
    --split-mode random
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Germano filtering and spectral restriction
# =============================================================================

def k2_numpy(n: int, lx: float, ly: float) -> np.ndarray:
    """Return kx^2 + ky^2 for a periodic n x n grid."""
    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=lx / n)
    ky = 2.0 * np.pi * np.fft.fftfreq(n, d=ly / n)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    return kx_grid**2 + ky_grid**2


def germano_filter(
    field: np.ndarray,
    alpha: float,
    lx: float,
    ly: float,
) -> np.ndarray:
    """
    Solve spectrally:

        (I - alpha^2 Laplacian) filtered = field
    """
    field_hat = np.fft.fft2(field)
    multiplier = 1.0 / (1.0 + alpha**2 * k2_numpy(field.shape[0], lx, ly))
    return np.fft.ifft2(multiplier * field_hat).real


def spectral_resize(field: np.ndarray, target_n: int) -> np.ndarray:
    """
    Periodic Fourier restriction or zero-padding for square even grids.
    """
    source_n = field.shape[0]

    if field.shape != (source_n, source_n):
        raise ValueError(f"Expected square field, received {field.shape}")

    source_hat = np.fft.fftshift(np.fft.fft2(field))
    target_hat = np.zeros((target_n, target_n), dtype=np.complex128)

    copy_n = min(source_n, target_n)
    source_start = (source_n - copy_n) // 2
    target_start = (target_n - copy_n) // 2

    target_hat[
        target_start : target_start + copy_n,
        target_start : target_start + copy_n,
    ] = source_hat[
        source_start : source_start + copy_n,
        source_start : source_start + copy_n,
    ]

    target_hat *= (target_n / source_n) ** 2

    return np.fft.ifft2(np.fft.ifftshift(target_hat)).real


def k2_torch(
    n: int,
    lx: float,
    ly: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    kx = 2.0 * math.pi * torch.fft.fftfreq(
        n,
        d=lx / n,
        device=device,
        dtype=dtype,
    )
    ky = 2.0 * math.pi * torch.fft.fftfreq(
        n,
        d=ly / n,
        device=device,
        dtype=dtype,
    )

    kx_grid, ky_grid = torch.meshgrid(kx, ky, indexing="ij")
    return kx_grid.square() + ky_grid.square()


def inverse_germano_torch(
    filtered: torch.Tensor,
    alpha: float,
    lx: float,
    ly: float,
) -> torch.Tensor:
    """
    Apply:

        field = (I - alpha^2 Laplacian) filtered

    Input shape: [batch, channels, n, n]
    """
    n = filtered.shape[-1]
    filtered_hat = torch.fft.fft2(filtered, dim=(-2, -1))
    multiplier = 1.0 + alpha**2 * k2_torch(
        n,
        lx,
        ly,
        filtered.device,
        filtered.dtype,
    )
    return torch.fft.ifft2(
        multiplier * filtered_hat,
        dim=(-2, -1),
    ).real


# =============================================================================
# Dataset discovery and splitting
# =============================================================================

def recursive_npz_files(directory: Path) -> list[Path]:
    return sorted(path for path in directory.rglob("*.npz") if path.is_file())


@dataclass
class DatasetSplit:
    train: list[Path]
    validation: list[Path]
    test: list[Path]


def discover_dataset(
    dataset_path: Path,
    validation_fraction: float,
    test_fraction: float,
    split_mode: str,
    seed: int,
) -> DatasetSplit:
    """
    Use train/validation/test subdirectories when all are present.
    Otherwise split all recursively discovered NPZ files.
    """
    train_dir = dataset_path / "train"
    validation_dir = dataset_path / "validation"
    test_dir = dataset_path / "test"

    if train_dir.is_dir() and validation_dir.is_dir() and test_dir.is_dir():
        train_files = recursive_npz_files(train_dir)
        validation_files = recursive_npz_files(validation_dir)
        test_files = recursive_npz_files(test_dir)

        if not train_files:
            raise ValueError(f"No NPZ files found in {train_dir}")
        if not validation_files:
            raise ValueError(f"No NPZ files found in {validation_dir}")
        if not test_files:
            raise ValueError(f"No NPZ files found in {test_dir}")

        return DatasetSplit(train_files, validation_files, test_files)

    files = recursive_npz_files(dataset_path)

    if len(files) < 3:
        raise ValueError(
            "At least three NPZ files are required when automatic splitting "
            "is used."
        )

    if validation_fraction <= 0.0 or test_fraction <= 0.0:
        raise ValueError("Validation and test fractions must be positive.")

    if validation_fraction + test_fraction >= 1.0:
        raise ValueError(
            "validation_fraction + test_fraction must be smaller than 1."
        )

    if split_mode == "random":
        generator = random.Random(seed)
        generator.shuffle(files)
    elif split_mode != "sequential":
        raise ValueError(f"Unsupported split mode: {split_mode}")

    n_total = len(files)
    n_test = max(1, round(test_fraction * n_total))
    n_validation = max(1, round(validation_fraction * n_total))
    n_train = n_total - n_validation - n_test

    if n_train < 1:
        raise ValueError("The requested split leaves no training snapshots.")

    return DatasetSplit(
        train=files[:n_train],
        validation=files[n_train : n_train + n_validation],
        test=files[n_train + n_validation :],
    )


# =============================================================================
# Snapshot loading
# =============================================================================

def load_dns_snapshot(
    path: Path,
    field_names: Sequence[str],
    expected_n: int,
) -> np.ndarray:
    """
    Return DNS with shape [channels, expected_n, expected_n].
    """
    with np.load(path) as source:
        missing = [name for name in field_names if name not in source]

        if missing:
            raise KeyError(f"{path} is missing fields: {missing}")

        arrays = [
            np.asarray(source[name], dtype=np.float32)
            for name in field_names
        ]

    expected_shape = (expected_n, expected_n)

    for name, array in zip(field_names, arrays):
        if array.shape != expected_shape:
            raise ValueError(
                f"{path}: field {name!r} has shape {array.shape}; "
                f"expected {expected_shape}."
            )

        if not np.all(np.isfinite(array)):
            raise ValueError(f"{path}: field {name!r} contains NaN or Inf.")

    return np.stack(arrays, axis=0)


def filter_snapshot(
    dns: np.ndarray,
    alpha: float,
    lx: float,
    ly: float,
    coarse_n: int,
) -> tuple[np.ndarray, np.ndarray]:
    filtered_fine = np.stack(
        [
            germano_filter(channel, alpha, lx, ly)
            for channel in dns
        ],
        axis=0,
    ).astype(np.float32)

    filtered_coarse = np.stack(
        [
            spectral_resize(channel, coarse_n)
            for channel in filtered_fine
        ],
        axis=0,
    ).astype(np.float32)

    return filtered_coarse, filtered_fine


# =============================================================================
# Training-only normalisation
# =============================================================================

def calculate_training_statistics(
    files: Sequence[Path],
    field_names: Sequence[str],
    fine_n: int,
    coarse_n: int,
    alpha: float,
    lx: float,
    ly: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-channel mean and standard deviation of filtered fine fields
    using only training snapshots.

    Uses accumulated sums to avoid storing the entire training dataset.
    """
    n_channels = len(field_names)
    total_sum = np.zeros(n_channels, dtype=np.float64)
    total_square_sum = np.zeros(n_channels, dtype=np.float64)
    total_points = 0

    print("Calculating training normalisation statistics...")

    for index, path in enumerate(files, start=1):
        dns = load_dns_snapshot(path, field_names, fine_n)
        _, filtered_fine = filter_snapshot(
            dns,
            alpha,
            lx,
            ly,
            coarse_n,
        )

        total_sum += filtered_fine.sum(axis=(1, 2), dtype=np.float64)
        total_square_sum += np.square(
            filtered_fine,
            dtype=np.float64,
        ).sum(axis=(1, 2))
        total_points += fine_n * fine_n

        if index == 1 or index % 100 == 0 or index == len(files):
            print(f"  statistics: {index}/{len(files)}")

    mean = total_sum / total_points
    variance = total_square_sum / total_points - mean**2
    variance = np.maximum(variance, 0.0)
    std = np.sqrt(variance)
    std = np.maximum(std, 1.0e-8)

    return (
        mean.astype(np.float32).reshape(1, n_channels, 1, 1),
        std.astype(np.float32).reshape(1, n_channels, 1, 1),
    )


# =============================================================================
# PyTorch dataset
# =============================================================================

class GermanoSnapshotDataset(Dataset):
    def __init__(
        self,
        files: Sequence[Path],
        field_names: Sequence[str],
        fine_n: int,
        coarse_n: int,
        alpha: float,
        lx: float,
        ly: float,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> None:
        self.files = list(files)
        self.field_names = tuple(field_names)
        self.fine_n = fine_n
        self.coarse_n = coarse_n
        self.alpha = alpha
        self.lx = lx
        self.ly = ly

        # Remove leading batch dimension: [1,C,1,1] -> [C,1,1].
        self.mean = mean[0]
        self.std = std[0]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        dns = load_dns_snapshot(
            self.files[index],
            self.field_names,
            self.fine_n,
        )

        filtered_coarse, filtered_fine = filter_snapshot(
            dns,
            self.alpha,
            self.lx,
            self.ly,
            self.coarse_n,
        )

        coarse_normalised = (filtered_coarse - self.mean) / self.std
        fine_normalised = (filtered_fine - self.mean) / self.std

        return (
            torch.from_numpy(coarse_normalised).float(),
            torch.from_numpy(fine_normalised).float(),
            torch.from_numpy(dns).float(),
        )


# =============================================================================
# Existing residual U-Net
# =============================================================================

class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        groups = min(8, out_channels)

        while out_channels % groups:
            groups -= 1

        self.network = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(groups, out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class UpBlock(nn.Module):
    def __init__(
        self,
        input_channels: int,
        skip_channels: int,
        output_channels: int,
    ) -> None:
        super().__init__()
        self.network = ConvBlock(
            input_channels + skip_channels,
            output_channels,
        )

    def forward(
        self,
        x: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        x = nn.functional.interpolate(
            x,
            size=skip.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        return self.network(torch.cat((x, skip), dim=1))


class ResidualUNetSR(nn.Module):
    """
    Input:  [batch, channels, 32, 32]
    Output: [batch, channels, 256, 256]

    The architecture is unchanged from the single-image test.
    """

    def __init__(
        self,
        channels: int = 3,
        base_channels: int = 24,
    ) -> None:
        super().__init__()

        base = base_channels
        self.pool = nn.AvgPool2d(2)

        self.encoder1 = ConvBlock(channels, base)
        self.encoder2 = ConvBlock(base, 2 * base)
        self.encoder3 = ConvBlock(2 * base, 4 * base)
        self.encoder4 = ConvBlock(4 * base, 8 * base)

        self.bottleneck = ConvBlock(8 * base, 8 * base)

        self.decoder3 = UpBlock(8 * base, 4 * base, 4 * base)
        self.decoder2 = UpBlock(4 * base, 2 * base, 2 * base)
        self.decoder1 = UpBlock(2 * base, base, base)

        self.output = nn.Conv2d(base, channels, 1)

    def forward(self, coarse: torch.Tensor) -> torch.Tensor:
        baseline = nn.functional.interpolate(
            coarse,
            scale_factor=8,
            mode="bicubic",
            align_corners=False,
        )

        encoder1 = self.encoder1(baseline)
        encoder2 = self.encoder2(self.pool(encoder1))
        encoder3 = self.encoder3(self.pool(encoder2))
        encoder4 = self.encoder4(self.pool(encoder3))

        bottleneck = self.bottleneck(encoder4)

        decoder3 = self.decoder3(bottleneck, encoder3)
        decoder2 = self.decoder2(decoder3, encoder2)
        decoder1 = self.decoder1(decoder2, encoder1)

        return baseline + self.output(decoder1)


# =============================================================================
# Losses and metrics: unchanged formulation
# =============================================================================

@dataclass
class LossValues:
    total: torch.Tensor
    filtered: torch.Tensor
    dns: torch.Tensor
    spectrum: torch.Tensor


def calculate_losses(
    model: nn.Module,
    coarse: torch.Tensor,
    filtered_target_normalised: torch.Tensor,
    dns_target: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    alpha: float,
    lx: float,
    ly: float,
    lambda_dns: float,
    lambda_spectrum: float,
) -> tuple[LossValues, torch.Tensor, torch.Tensor]:
    predicted_normalised = model(coarse)
    predicted_filtered = predicted_normalised * std + mean
    predicted_dns = inverse_germano_torch(
        predicted_filtered,
        alpha,
        lx,
        ly,
    )

    filtered_loss = nn.functional.l1_loss(
        predicted_normalised,
        filtered_target_normalised,
    )

    dns_loss = nn.functional.l1_loss(
        predicted_dns,
        dns_target,
    )

    predicted_hat = torch.fft.rfft2(
        predicted_normalised,
        dim=(-2, -1),
    )
    target_hat = torch.fft.rfft2(
        filtered_target_normalised,
        dim=(-2, -1),
    )

    spectrum_loss = nn.functional.l1_loss(
        torch.log1p(torch.abs(predicted_hat)),
        torch.log1p(torch.abs(target_hat)),
    )

    total_loss = (
        filtered_loss
        + lambda_dns * dns_loss
        + lambda_spectrum * spectrum_loss
    )

    return (
        LossValues(
            total=total_loss,
            filtered=filtered_loss,
            dns=dns_loss,
            spectrum=spectrum_loss,
        ),
        predicted_filtered,
        predicted_dns,
    )


def relative_l2_torch(
    reference: torch.Tensor,
    prediction: torch.Tensor,
) -> torch.Tensor:
    numerator = torch.linalg.vector_norm(
        (prediction - reference).flatten(1),
        dim=1,
    )
    denominator = torch.linalg.vector_norm(
        reference.flatten(1),
        dim=1,
    ).clamp_min(1.0e-12)

    return (numerator / denominator).mean()


# =============================================================================
# Training and evaluation loops
# =============================================================================

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
    alpha: float,
    lx: float,
    ly: float,
    lambda_dns: float,
    lambda_spectrum: float,
) -> dict[str, float]:
    model.train()

    totals = {
        "total": 0.0,
        "filtered": 0.0,
        "dns": 0.0,
        "spectrum": 0.0,
    }
    sample_count = 0

    for coarse, filtered_target, dns_target in loader:
        coarse = coarse.to(device, non_blocking=True)
        filtered_target = filtered_target.to(device, non_blocking=True)
        dns_target = dns_target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        losses, _, _ = calculate_losses(
            model,
            coarse,
            filtered_target,
            dns_target,
            mean,
            std,
            alpha,
            lx,
            ly,
            lambda_dns,
            lambda_spectrum,
        )

        losses.total.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        batch_size = coarse.shape[0]
        sample_count += batch_size
        totals["total"] += float(losses.total.detach()) * batch_size
        totals["filtered"] += float(losses.filtered.detach()) * batch_size
        totals["dns"] += float(losses.dns.detach()) * batch_size
        totals["spectrum"] += float(losses.spectrum.detach()) * batch_size

    return {
        key: value / sample_count
        for key, value in totals.items()
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
    alpha: float,
    lx: float,
    ly: float,
    lambda_dns: float,
    lambda_spectrum: float,
) -> dict[str, float]:
    model.eval()

    totals = {
        "total": 0.0,
        "filtered": 0.0,
        "dns": 0.0,
        "spectrum": 0.0,
        "filtered_relative_l2": 0.0,
        "dns_relative_l2": 0.0,
    }
    sample_count = 0

    for coarse, filtered_target, dns_target in loader:
        coarse = coarse.to(device, non_blocking=True)
        filtered_target = filtered_target.to(device, non_blocking=True)
        dns_target = dns_target.to(device, non_blocking=True)

        losses, predicted_filtered, predicted_dns = calculate_losses(
            model,
            coarse,
            filtered_target,
            dns_target,
            mean,
            std,
            alpha,
            lx,
            ly,
            lambda_dns,
            lambda_spectrum,
        )

        target_filtered = filtered_target * std + mean
        filtered_relative_l2 = relative_l2_torch(
            target_filtered,
            predicted_filtered,
        )
        dns_relative_l2 = relative_l2_torch(
            dns_target,
            predicted_dns,
        )

        batch_size = coarse.shape[0]
        sample_count += batch_size

        totals["total"] += float(losses.total) * batch_size
        totals["filtered"] += float(losses.filtered) * batch_size
        totals["dns"] += float(losses.dns) * batch_size
        totals["spectrum"] += float(losses.spectrum) * batch_size
        totals["filtered_relative_l2"] += (
            float(filtered_relative_l2) * batch_size
        )
        totals["dns_relative_l2"] += float(dns_relative_l2) * batch_size

    return {
        key: value / sample_count
        for key, value in totals.items()
    }


# =============================================================================
# Spectrum and plotting
# =============================================================================

def isotropic_spectrum(
    field: np.ndarray,
    lx: float,
    ly: float,
) -> tuple[np.ndarray, np.ndarray]:
    n = field.shape[0]

    kx = 2.0 * np.pi * np.fft.fftfreq(n, d=lx / n)
    ky = 2.0 * np.pi * np.fft.fftfreq(n, d=ly / n)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    magnitude = np.sqrt(kx_grid**2 + ky_grid**2)

    field_hat = np.fft.fft2(field)
    modal_power = np.abs(field_hat) ** 2 / n**4

    delta_k = min(2.0 * np.pi / lx, 2.0 * np.pi / ly)
    shell = np.rint(magnitude / delta_k).astype(np.int64)

    power = np.bincount(shell.ravel(), weights=modal_power.ravel())
    counts = np.bincount(shell.ravel())
    wavenumber = np.arange(power.size) * delta_k

    valid = counts > 0
    return wavenumber[valid], power[valid]


def save_image(
    field: np.ndarray,
    title: str,
    path: Path,
    lx: float,
    ly: float,
) -> None:
    fig, axis = plt.subplots(figsize=(6.3, 5.2))

    image = axis.imshow(
        field.T,
        origin="lower",
        extent=(0.0, lx, 0.0, ly),
        aspect="equal",
    )

    axis.set_title(title)
    axis.set_xlabel("x")
    axis.set_ylabel("y")
    fig.colorbar(image, ax=axis)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_spectrum_plot(
    field_name: str,
    dns: np.ndarray,
    filtered_fine: np.ndarray,
    filtered_coarse: np.ndarray,
    predicted_filtered: np.ndarray,
    predicted_dns: np.ndarray,
    coarse_n: int,
    lx: float,
    ly: float,
    path: Path,
) -> None:
    spectra = {
        "DNS": isotropic_spectrum(dns, lx, ly),
        "Filtered 256": isotropic_spectrum(filtered_fine, lx, ly),
        "Filtered 32": isotropic_spectrum(filtered_coarse, lx, ly),
        "U-Net filtered 256": isotropic_spectrum(
            predicted_filtered,
            lx,
            ly,
        ),
        "Reconstructed DNS": isotropic_spectrum(predicted_dns, lx, ly),
    }

    fig, axis = plt.subplots(figsize=(7.4, 5.6))

    for label, (wavenumber, power) in spectra.items():
        valid = (wavenumber > 0.0) & (power > 0.0)

        if label == "Filtered 32":
            axis.loglog(
                wavenumber[valid],
                power[valid],
                linestyle="None",
                marker="o",
                markersize=4,
                label=label,
            )
        elif label == "U-Net filtered 256":
            axis.loglog(
                wavenumber[valid],
                power[valid],
                linestyle="--",
                linewidth=1.8,
                label=label,
            )
        else:
            axis.loglog(
                wavenumber[valid],
                power[valid],
                linewidth=1.8,
                label=label,
            )

    coarse_nyquist = min(
        np.pi / (lx / coarse_n),
        np.pi / (ly / coarse_n),
    )

    axis.axvline(
        coarse_nyquist,
        linestyle=":",
        linewidth=1.4,
        label=f"{coarse_n}-grid Nyquist",
    )

    axis.set_title(f"{field_name}: isotropic spectrum comparison")
    axis.set_xlabel(r"$k$")
    axis.set_ylabel(r"Shell power $P(k)$")
    axis.set_ylim(bottom=1.0e-10)
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(numpoints=1)

    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


@torch.no_grad()
def save_test_example(
    model: nn.Module,
    dataset: GermanoSnapshotDataset,
    sample_index: int,
    device: torch.device,
    mean: torch.Tensor,
    std: torch.Tensor,
    alpha: float,
    lx: float,
    ly: float,
    field_names: Sequence[str],
    coarse_n: int,
    output_dir: Path,
) -> None:
    model.eval()

    coarse, filtered_target_normalised, dns_target = dataset[sample_index]

    coarse_batch = coarse.unsqueeze(0).to(device)
    filtered_target_batch = filtered_target_normalised.unsqueeze(0).to(device)
    dns_target_batch = dns_target.unsqueeze(0).to(device)

    _, predicted_filtered, predicted_dns = calculate_losses(
        model,
        coarse_batch,
        filtered_target_batch,
        dns_target_batch,
        mean,
        std,
        alpha,
        lx,
        ly,
        lambda_dns=0.0,
        lambda_spectrum=0.0,
    )

    target_filtered = filtered_target_batch * std + mean

    coarse_np = coarse.cpu().numpy()
    target_filtered_np = target_filtered[0].cpu().numpy()
    dns_np = dns_target.cpu().numpy()
    predicted_filtered_np = predicted_filtered[0].cpu().numpy()
    predicted_dns_np = predicted_dns[0].cpu().numpy()

    np.savez_compressed(
        output_dir / "test_example_results.npz",
        filtered_32=coarse_np,
        filtered_256=target_filtered_np,
        dns_256=dns_np,
        unet_filtered_256=predicted_filtered_np,
        reconstructed_dns_256=predicted_dns_np,
        field_names=np.asarray(field_names),
    )

    for channel, field_name in enumerate(field_names):
        save_image(
            predicted_filtered_np[channel],
            f"{field_name}: U-Net reconstructed filtered field",
            output_dir / f"{field_name}_unet_filtered_256.png",
            lx,
            ly,
        )

        save_image(
            predicted_dns_np[channel],
            f"{field_name}: inverse-filtered U-Net reconstruction",
            output_dir / f"{field_name}_unet_reconstructed_dns_256.png",
            lx,
            ly,
        )

        save_image(
            predicted_dns_np[channel] - dns_np[channel],
            f"{field_name}: DNS reconstruction error",
            output_dir / f"{field_name}_unet_dns_error.png",
            lx,
            ly,
        )

        save_spectrum_plot(
            field_name,
            dns_np[channel],
            target_filtered_np[channel],
            # Convert normalised coarse field back to physical units.
            coarse_np[channel] * float(std[0, channel, 0, 0].cpu())
            + float(mean[0, channel, 0, 0].cpu()),
            predicted_filtered_np[channel],
            predicted_dns_np[channel],
            coarse_n,
            lx,
            ly,
            output_dir / f"{field_name}_unet_spectrum_comparison.png",
        )


# =============================================================================
# Command line and main
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Germano U-Net on a directory of NPZ snapshots."
    )

    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help=(
            "Dataset directory. Either provide train/validation/test "
            "subdirectories or one directory of NPZ files."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("germano_unet_dataset_output"),
    )

    parser.add_argument(
        "--fields",
        nargs=3,
        default=["U", "V", "P"],
    )

    parser.add_argument("--fine-n", type=int, default=256)
    parser.add_argument("--coarse-n", type=int, default=32)
    parser.add_argument("--lx", type=float, default=2.0 * np.pi)
    parser.add_argument("--ly", type=float, default=2.0 * np.pi)

    alpha_group = parser.add_mutually_exclusive_group()
    alpha_group.add_argument("--alpha", type=float, default=None)
    alpha_group.add_argument("--alpha-factor", type=float, default=1.0)

    parser.add_argument(
        "--split-mode",
        choices=["sequential", "random"],
        default="sequential",
        help=(
            "Used only when train/validation/test directories are absent. "
            "Sequential is safer for time-correlated DNS data."
        ),
    )
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.1)

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2.0e-4)
    parser.add_argument("--base-channels", type=int, default=24)
    parser.add_argument("--lambda-dns", type=float, default=0.05)
    parser.add_argument("--lambda-spectrum", type=float, default=0.02)

    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--print-every", type=int, default=1)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)

    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    set_seed(arguments.seed)

    if not arguments.dataset.is_dir():
        raise NotADirectoryError(arguments.dataset)

    if arguments.fine_n != 8 * arguments.coarse_n:
        raise ValueError(
            "The current unchanged architecture requires fine_n = 8*coarse_n."
        )

    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = arguments.output_dir / "checkpoints"
    checkpoints_dir.mkdir(exist_ok=True)

    split = discover_dataset(
        arguments.dataset,
        arguments.validation_fraction,
        arguments.test_fraction,
        arguments.split_mode,
        arguments.seed,
    )

    delta = arguments.lx / arguments.fine_n
    alpha = (
        arguments.alpha
        if arguments.alpha is not None
        else arguments.alpha_factor * delta
    )

    print("Dataset")
    print(f"  train      : {len(split.train)}")
    print(f"  validation : {len(split.validation)}")
    print(f"  test       : {len(split.test)}")
    print(f"  alpha      : {alpha:.12g}")
    print(f"  alpha/delta: {alpha / delta:.6g}")

    mean_numpy, std_numpy = calculate_training_statistics(
        split.train,
        arguments.fields,
        arguments.fine_n,
        arguments.coarse_n,
        alpha,
        arguments.lx,
        arguments.ly,
    )

    train_dataset = GermanoSnapshotDataset(
        split.train,
        arguments.fields,
        arguments.fine_n,
        arguments.coarse_n,
        alpha,
        arguments.lx,
        arguments.ly,
        mean_numpy,
        std_numpy,
    )
    validation_dataset = GermanoSnapshotDataset(
        split.validation,
        arguments.fields,
        arguments.fine_n,
        arguments.coarse_n,
        alpha,
        arguments.lx,
        arguments.ly,
        mean_numpy,
        std_numpy,
    )
    test_dataset = GermanoSnapshotDataset(
        split.test,
        arguments.fields,
        arguments.fine_n,
        arguments.coarse_n,
        alpha,
        arguments.lx,
        arguments.ly,
        mean_numpy,
        std_numpy,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=arguments.batch_size,
        shuffle=True,
        num_workers=arguments.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=arguments.num_workers > 0,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=arguments.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=arguments.num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=arguments.batch_size,
        shuffle=False,
        num_workers=arguments.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=arguments.num_workers > 0,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = ResidualUNetSR(
        channels=len(arguments.fields),
        base_channels=arguments.base_channels,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=arguments.learning_rate,
        weight_decay=1.0e-6,
    )

    mean = torch.from_numpy(mean_numpy).to(device)
    std = torch.from_numpy(std_numpy).to(device)

    best_validation = float("inf")
    history: list[dict[str, float | int]] = []

    best_checkpoint = checkpoints_dir / "best.pt"
    latest_checkpoint = checkpoints_dir / "latest.pt"

    for epoch in range(1, arguments.epochs + 1):
        training_metrics = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            mean,
            std,
            alpha,
            arguments.lx,
            arguments.ly,
            arguments.lambda_dns,
            arguments.lambda_spectrum,
        )

        validation_metrics = evaluate(
            model,
            validation_loader,
            device,
            mean,
            std,
            alpha,
            arguments.lx,
            arguments.ly,
            arguments.lambda_dns,
            arguments.lambda_spectrum,
        )

        row = {
            "epoch": epoch,
            "train_total": training_metrics["total"],
            "train_filtered": training_metrics["filtered"],
            "train_dns": training_metrics["dns"],
            "train_spectrum": training_metrics["spectrum"],
            "validation_total": validation_metrics["total"],
            "validation_filtered": validation_metrics["filtered"],
            "validation_dns": validation_metrics["dns"],
            "validation_spectrum": validation_metrics["spectrum"],
            "validation_filtered_relative_l2": validation_metrics[
                "filtered_relative_l2"
            ],
            "validation_dns_relative_l2": validation_metrics[
                "dns_relative_l2"
            ],
        }
        history.append(row)

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "validation_loss": validation_metrics["total"],
            "alpha": alpha,
            "delta": delta,
            "mean": mean_numpy,
            "std": std_numpy,
            "field_names": list(arguments.fields),
            "fine_n": arguments.fine_n,
            "coarse_n": arguments.coarse_n,
            "lx": arguments.lx,
            "ly": arguments.ly,
        }

        torch.save(checkpoint, latest_checkpoint)

        if validation_metrics["total"] < best_validation:
            best_validation = validation_metrics["total"]
            torch.save(checkpoint, best_checkpoint)

        if arguments.save_every > 0 and epoch % arguments.save_every == 0:
            torch.save(
                checkpoint,
                checkpoints_dir / f"epoch_{epoch:05d}.pt",
            )

        if epoch == 1 or epoch % arguments.print_every == 0:
            print(
                f"Epoch {epoch:05d} | "
                f"train={training_metrics['total']:.6e} | "
                f"validation={validation_metrics['total']:.6e} | "
                f"validation DNS relL2="
                f"{validation_metrics['dns_relative_l2']:.6e}"
            )

        with (arguments.output_dir / "training_history.csv").open(
            "w",
            newline="",
        ) as history_file:
            writer = csv.DictWriter(
                history_file,
                fieldnames=list(history[0].keys()),
            )
            writer.writeheader()
            writer.writerows(history)

    best = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(best["model"])

    test_metrics = evaluate(
        model,
        test_loader,
        device,
        mean,
        std,
        alpha,
        arguments.lx,
        arguments.ly,
        arguments.lambda_dns,
        arguments.lambda_spectrum,
    )

    save_test_example(
        model,
        test_dataset,
        sample_index=0,
        device=device,
        mean=mean,
        std=std,
        alpha=alpha,
        lx=arguments.lx,
        ly=arguments.ly,
        field_names=arguments.fields,
        coarse_n=arguments.coarse_n,
        output_dir=arguments.output_dir,
    )

    metadata = {
        "dataset": str(arguments.dataset),
        "train_snapshots": len(split.train),
        "validation_snapshots": len(split.validation),
        "test_snapshots": len(split.test),
        "field_names": list(arguments.fields),
        "fine_n": arguments.fine_n,
        "coarse_n": arguments.coarse_n,
        "lx": arguments.lx,
        "ly": arguments.ly,
        "alpha": alpha,
        "delta": delta,
        "alpha_over_delta": alpha / delta,
        "best_epoch": int(best["epoch"]),
        "best_validation_loss": float(best["validation_loss"]),
        "test_metrics": test_metrics,
    }

    (arguments.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )

    print("\nTest metrics")
    for name, value in test_metrics.items():
        print(f"  {name:24s}: {value:.6e}")

    print(f"\nBest checkpoint: {best_checkpoint}")
    print(f"Results:         {arguments.output_dir}")


if __name__ == "__main__":
    main()
