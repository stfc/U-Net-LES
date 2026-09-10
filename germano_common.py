#!/usr/bin/env python3
"""Common components for Germano-filter U-Net training and inference."""

from __future__ import annotations
import math, os, random
from pathlib import Path
from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

SPECTRUM_Y_MIN = float(os.environ.get("SPECTRUM_Y_MIN", "1e-14"))
if not math.isfinite(SPECTRUM_Y_MIN) or SPECTRUM_Y_MIN <= 0:
    raise ValueError("SPECTRUM_Y_MIN must be finite and greater than zero")
SPECTRUM_Y_MAX = 10.0

HW_FIELD_NAMES = ("U", "V", "P")
HW_FIELD_LABELS = \
{
    "U": r"electron density $n_e$",
    "V": r"electrostatic potential $\phi$",
    "P": r"vorticity $\omega$",
}


def rescaling_amplitude_correction \
(
    les: torch.Tensor, filtered_dns: torch.Tensor,
) -> torch.Tensor:
    """Return the median equation-11 correction per batch/channel.

    For DNS amplitude A and LES amplitude L, equation 11 gives
    A_new = median((LES / L) / (G(DNS) / A)) * L.
    Hence A_new / A = median(LES / G(DNS)); normalization cancels.
    Inputs must share a grid. Exclude nonfinite ratios and denominators
    within machine precision of zero. Retain the amplitude if no valid
    ratio exists; an identically zero LES requests zero amplitude.
    """
    if les.shape != filtered_dns.shape or les.ndim != 4:
        raise ValueError("Rescaling requires matching (batch, channel, x, y) fields")
    finite_magnitude = torch.where \
    (
        torch.isfinite(filtered_dns), filtered_dns.abs(), torch.zeros_like(filtered_dns),
    )
    magnitude = finite_magnitude.amax(dim=(-2, -1), keepdim=True)
    tolerance = torch.finfo(filtered_dns.dtype).eps * magnitude
    valid = torch.isfinite(les) & torch.isfinite(filtered_dns)
    valid = valid & (filtered_dns.abs() > tolerance)
    denominator = torch.where(valid, filtered_dns, torch.ones_like(filtered_dns))
    ratios = les / denominator
    valid = valid & torch.isfinite(ratios)
    ratios = torch.where(valid, ratios, torch.full_like(ratios, float("nan")))
    # Average the two middle values for an even-sized distribution.
    correction = torch.nanquantile \
    (
        ratios.flatten(-2), 0.5, dim=-1, keepdim=True,
    ).unsqueeze(-1)
    correction = torch.where \
    (
        torch.isfinite(correction), correction, torch.ones_like(correction),
    )
    return torch.where \
    (
        (les == 0).all(dim=-1, keepdim=True).all(dim=-2, keepdim=True),
        torch.zeros_like(correction), correction,
    )


def rescaling_amplitude_correction13 \
(
    dns: torch.Tensor, les: torch.Tensor,
    previous_dns_max: torch.Tensor, previous_les_max: torch.Tensor,
) -> torch.Tensor:
    """Equation 13: DNS_max_new = LES_max_new * DNS_max_old / LES_max_old.

    Use absolute maxima for signed fields. Initial maxima are equivalent
    to previous-step maxima because this recurrence preserves their ratio.
    Retain the amplitude for a zero reference LES or zero reconstruction;
    an identically zero current LES requests zero amplitude.
    """
    if dns.ndim != 4 or les.ndim != 4 or dns.shape[:2] != les.shape[:2]:
        raise ValueError("Rescaling requires matching batch/channel dimensions")
    for maximum in (previous_dns_max, previous_les_max):
        if maximum.shape not in ((1, dns.shape[1], 1, 1), dns.shape[:2] + (1, 1)):
            raise ValueError("Rescaling maxima must have shape (batch, channel, 1, 1)")
        if not torch.isfinite(maximum).all() or (maximum < 0).any():
            raise ValueError("Rescaling maxima must be finite and nonnegative")
    dns_max = dns.abs().amax(dim=(-2, -1), keepdim=True)
    les_max = les.abs().amax(dim=(-2, -1), keepdim=True)
    reference_denominator = torch.where \
    (
        previous_les_max > 0, previous_les_max, torch.ones_like(previous_les_max),
    )
    dns_denominator = torch.where(dns_max > 0, dns_max, torch.ones_like(dns_max))
    target_max = les_max * (previous_dns_max / reference_denominator)
    correction = torch.where \
    (
        (previous_les_max > 0) & (dns_max > 0),
        target_max / dns_denominator, torch.ones_like(dns_max),
    )
    return torch.where(les_max == 0, torch.zeros_like(correction), correction)


def load_checkpoint(path: Path, device: torch.device):
    """Load a full checkpoint produced by this project.

    These checkpoints contain optimizer state and NumPy arrays in addition to
    model weights, so PyTorch 2.6's weights-only loader cannot deserialize them.
    Only call this for checkpoints from a trusted source.
    """
    return torch.load(path, map_location=device, weights_only=False)


def validate_hw_checkpoint(checkpoint: dict) -> None:
    fields = tuple(checkpoint.get("field_names", ()))
    if fields != HW_FIELD_NAMES or checkpoint.get("physics_version") != 1:
        raise ValueError \
        (
            "This checkpoint predates the Hasegawa-Wakatani constraints. "
            "Retrain it with fields U, V, P before inference or resuming."
        )
    mode = checkpoint.get("vorticity_mode")
    if mode not in ("learned_channel", "laplacian_of_potential"):
        raise ValueError \
        (
            "Checkpoint vorticity mode must be learned_channel or "
            "laplacian_of_potential."
        )
    expected_outputs = 3 if mode == "learned_channel" else 2
    if checkpoint.get("model_output_channels") != expected_outputs:
        raise ValueError \
        (
            f"{mode} checkpoints require {expected_outputs} model outputs."
        )
    validate_grid_pair \
    (
        int(checkpoint["fine_n"]),
        int(checkpoint["coarse_n"]),
    )


def validate_grid_pair(fine_n: int, coarse_n: int) -> int:
    """Validate a DNS/LES grid pair and return its refinement ratio."""
    if fine_n <= 0 or coarse_n <= 0:
        raise ValueError("DNS fine_n and LES coarse_n must be positive.")
    if coarse_n > fine_n:
        raise ValueError("LES coarse_n cannot exceed DNS fine_n.")
    if fine_n % coarse_n:
        raise ValueError \
        (
            "DNS fine_n must be an integer multiple of LES coarse_n."
        )
    if fine_n % 8:
        raise ValueError \
        (
            "DNS fine_n must be divisible by 8 for the three-level U-Net."
        )
    return fine_n // coarse_n


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def k2_numpy(n: int, lx: float, ly: float) -> np.ndarray:
    kx = 2 * np.pi * np.fft.fftfreq(n, d=lx / n)
    ky = 2 * np.pi * np.fft.fftfreq(n, d=ly / n)
    kxg, kyg = np.meshgrid(kx, ky, indexing="ij")
    return kxg**2 + kyg**2


def germano_filter \
(
    field: np.ndarray, alpha: float, lx: float, ly: float
) -> np.ndarray:
    fh = np.fft.fft2(field)
    return np.fft.ifft2 \
    (
        fh / (1 + alpha**2 * k2_numpy(field.shape[0], lx, ly))
    ).real


def spectral_resize(field: np.ndarray, target_n: int) -> np.ndarray:
    source_n = field.shape[0]
    if field.shape != (source_n, source_n):
        raise ValueError(f"Expected square field, received {field.shape}")
    sh = np.fft.fftshift(np.fft.fft2(field))
    th = np.zeros((target_n, target_n), complex)
    copy_n = min(source_n, target_n)
    ss = (source_n - copy_n) // 2
    ts = (target_n - copy_n) // 2
    th[ts : ts + copy_n, ts : ts + copy_n] = sh \
    [
        ss : ss + copy_n, ss : ss + copy_n
    ]
    th *= (target_n / source_n) ** 2
    return np.fft.ifft2(np.fft.ifftshift(th)).real


def k2_torch \
(
    n: int, lx: float, ly: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    kx = \
    (
        2 * math.pi * torch.fft.fftfreq(n, d=lx / n, device=device, dtype=dtype)
    )
    ky = \
    (
        2 * math.pi * torch.fft.fftfreq(n, d=ly / n, device=device, dtype=dtype)
    )
    kxg, kyg = torch.meshgrid(kx, ky, indexing="ij")
    return kxg.square() + kyg.square()


def inverse_germano_torch \
(
    filtered: torch.Tensor,
    alpha: float,
    lx: float,
    ly: float,
    cutoff_fraction=None,
    cutoff_order=8,
) -> torch.Tensor:
    n = filtered.shape[-1]
    fh = torch.fft.fft2(filtered, dim=(-2, -1))
    k2 = k2_torch(n, lx, ly, filtered.device, filtered.dtype)
    mult = 1 + alpha**2 * k2
    if cutoff_fraction is not None:
        if not 0 < cutoff_fraction <= 1:
            raise ValueError("cutoff_fraction must be in (0, 1].")
        if cutoff_order <= 0:
            raise ValueError("cutoff_order must be positive.")
        grid_nyquist = min(math.pi / (lx / n), math.pi / (ly / n))
        cutoff_wavenumber = cutoff_fraction * grid_nyquist
        mult = mult * torch.exp \
        (
            -(torch.sqrt(k2) / cutoff_wavenumber) ** cutoff_order
        )
    reconstructed = torch.fft.ifft2(mult * fh, dim=(-2, -1)).real
    return enforce_zero_spatial_mean(reconstructed)


INVERSE_METHODS = ("exact", "van_cittert")


def van_cittert_torch \
(
    filtered: torch.Tensor,
    alpha: float,
    lx: float,
    ly: float,
    iterations: int = 1,
    relaxation: float = 1.0,
) -> torch.Tensor:
    """Approximately invert the Germano filter with relaxed van Cittert.

    ``filtered`` is kept as the fixed right-hand side in
    ``u[n+1] = u[n] + relaxation * (filtered - G(u[n]))``.  Starting from
    ``u[0] = filtered`` makes ``iterations=0`` the identity reconstruction.
    """
    if isinstance(iterations, bool) or not isinstance(iterations, int):
        raise TypeError("iterations must be an integer.")
    if iterations < 0:
        raise ValueError("iterations must be non-negative.")
    if not math.isfinite(relaxation) or not 0 < relaxation <= 1:
        raise ValueError("relaxation must be finite and in (0, 1].")

    n = filtered.shape[-1]
    k2 = k2_torch(n, lx, ly, filtered.device, filtered.dtype)
    filter_multiplier = 1 / (1 + alpha**2 * k2)
    estimate = filtered
    for _ in range(iterations):
        estimate_hat = torch.fft.fft2(estimate, dim=(-2, -1))
        refiltered = torch.fft.ifft2 \
        (
            filter_multiplier * estimate_hat, dim=(-2, -1)
        ).real
        estimate = estimate + relaxation * (filtered - refiltered)
    return enforce_zero_spatial_mean(estimate)


def reconstruct_germano_torch \
(
    filtered: torch.Tensor,
    alpha: float,
    lx: float,
    ly: float,
    method: str = "exact",
    van_cittert_iterations: int = 1,
    van_cittert_relaxation: float = 1.0,
) -> torch.Tensor:
    """Reconstruct a Germano-filtered field with the selected inverse."""
    if method == "exact":
        return inverse_germano_torch(filtered, alpha, lx, ly)
    if method == "van_cittert":
        return van_cittert_torch \
        (
            filtered,
            alpha,
            lx,
            ly,
            van_cittert_iterations,
            van_cittert_relaxation,
        )
    raise ValueError \
    (
        f"Unknown inverse method {method!r}; choose from {INVERSE_METHODS}."
    )


def enforce_zero_spatial_mean(fields: torch.Tensor) -> torch.Tensor:
    """Project every batch/channel field onto the zero-spatial-mean subspace."""
    return fields - fields.mean(dim=(-2, -1), keepdim=True)


def restore_channel_spatial_mean \
(
    fields: torch.Tensor,
    reference: torch.Tensor,
    channel_index: int,
) -> torch.Tensor:
    """Give one channel the spatial mean of its reference field."""
    if fields.shape[0] != reference.shape[0]:
        raise ValueError("Fields and reference must have the same batch size.")
    if not 0 <= channel_index < fields.shape[1]:
        raise IndexError(f"Channel index {channel_index} is out of range.")
    reference_mean = reference[:, channel_index:channel_index + 1].mean \
    (
        dim=(-2, -1), keepdim=True
    )
    channels = list(fields.split(1, dim=1))
    channel = channels[channel_index]
    channels[channel_index] = \
    (
        channel - channel.mean(dim=(-2, -1), keepdim=True) + reference_mean
    )
    return torch.cat(channels, dim=1)


def spectral_cutoff_torch \
(
    field: torch.Tensor,
    lx: float,
    ly: float,
    cutoff_fraction,
    cutoff_order=8,
) -> torch.Tensor:
    """Apply a smooth exponential low-pass cutoff to a periodic field."""
    if cutoff_fraction is None:
        return field
    if not 0 < cutoff_fraction <= 1:
        raise ValueError("cutoff_fraction must be in (0, 1].")
    if cutoff_order <= 0:
        raise ValueError("cutoff_order must be positive.")
    n = field.shape[-1]
    k2 = k2_torch(n, lx, ly, field.device, field.dtype)
    grid_nyquist = min(math.pi / (lx / n), math.pi / (ly / n))
    cutoff_wavenumber = cutoff_fraction * grid_nyquist
    multiplier = torch.exp \
    (
        -(torch.sqrt(k2) / cutoff_wavenumber) ** cutoff_order
    )
    fh = torch.fft.fft2(field, dim=(-2, -1))
    return torch.fft.ifft2(multiplier * fh, dim=(-2, -1)).real


INVERSE_FILTER_TYPES = ("cutoff", "filtered_dns_spectrum")


def _spectral_shell_indices(n, lx, ly, device):
    """Return rounded isotropic-shell indices for an ``n x n`` grid."""
    kx = 2 * math.pi * torch.fft.fftfreq(n, d=lx / n, device=device)
    ky = 2 * math.pi * torch.fft.fftfreq(n, d=ly / n, device=device)
    dk = min(2 * math.pi / lx, 2 * math.pi / ly)
    return torch.round \
    (
        torch.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2) / dk
    ).long()


def filtered_dns_spectrum_filter_torch \
(
    field, lx, ly, target_shell_power, min_wavenumber=1.0
):
    """Suppress Fourier shells whose power exceeds the training-DNS mean.

    ``target_shell_power`` contains the mean power per Fourier mode for each
    output channel and radial shell.  A single multiplier is used per shell,
    preserving phases and relative modal structure.  The multiplier is capped
    at one, so this operation can only remove excess energy.  Energy-containing
    modes at ``k <= min_wavenumber`` pass through unchanged; the learned DNS
    spectrum therefore regularizes only the smaller scales.
    """
    if target_shell_power is None:
        raise ValueError \
        (
            "The filtered-DNS spectrum filter requires a spectrum stored in "
            "the checkpoint."
        )
    if not math.isfinite(min_wavenumber) or min_wavenumber < 0:
        raise ValueError("min_wavenumber must be finite and non-negative.")
    target = torch.as_tensor \
    (
        target_shell_power, dtype=field.dtype, device=field.device
    )
    if target.ndim != 2 or target.shape[0] != field.shape[1]:
        raise ValueError \
        (
            "Filtered-DNS target spectrum must have shape (channels, shells)."
        )
    fh = torch.fft.fft2(field, dim=(-2, -1))
    shells = _spectral_shell_indices(field.shape[-1], lx, ly, field.device)
    flat_shells = shells.flatten()
    shell_count = target.shape[1]
    if int(flat_shells.max()) >= shell_count:
        raise ValueError("Stored filtered-DNS spectrum has too few shells.")
    power = fh.abs().square().mean(dim=0).flatten(1)
    sums = torch.zeros \
    (
        (field.shape[1], shell_count), dtype=field.dtype,
        device=field.device
    )
    sums.scatter_add_(1, flat_shells.expand(field.shape[1], -1), power)
    counts = torch.bincount(flat_shells, minlength=shell_count).to(field.dtype)
    current = sums / counts.clamp_min(1).unsqueeze(0)
    scale = torch.sqrt \
    (
        target / current.clamp_min(torch.finfo(field.dtype).tiny)
    )
    scale = scale.clamp(max=1.0)
    multiplier = scale[:, flat_shells].reshape(1, field.shape[1], *shells.shape)
    if min_wavenumber > 0:
        wavenumber = torch.sqrt \
        (
            k2_torch \
            (
                field.shape[-1], lx, ly, field.device, field.dtype
            )
        )
        multiplier = torch.where \
        (
            wavenumber[None, None] <= min_wavenumber,
            torch.ones_like(multiplier),
            multiplier,
        )
    return torch.fft.ifft2(fh * multiplier, dim=(-2, -1)).real


def inverse_input_filter_torch \
(
    field, lx, ly, filter_type="cutoff", cutoff_fraction=None,
    cutoff_order=8, target_shell_power=None, min_wavenumber=1.0,
):
    """Apply the selected regularization before inverse reconstruction."""
    if filter_type not in INVERSE_FILTER_TYPES:
        raise ValueError(f"Unknown inverse filter {filter_type!r}.")
    if filter_type == "cutoff":
        return spectral_cutoff_torch \
        (
            field, lx, ly, cutoff_fraction, cutoff_order
        )
    return filtered_dns_spectrum_filter_torch \
    (
        field, lx, ly, target_shell_power, min_wavenumber=min_wavenumber
    )


POST_FILTER_TYPES = ("none", "high_order", "gaussian")


def post_filter_torch \
(
    field: torch.Tensor,
    lx: float,
    ly: float,
    filter_type="high_order",
    cutoff_fraction=None,
    cutoff_order=8,
) -> torch.Tensor:
    """Apply a selected post-inversion spectral low-pass filter."""
    if filter_type not in POST_FILTER_TYPES:
        raise ValueError \
        (
            f"Unknown post filter {filter_type!r}; choose from "
            f"{POST_FILTER_TYPES}."
        )
    if filter_type == "none":
        return field
    if filter_type == "high_order":
        return spectral_cutoff_torch \
        (
            field, lx, ly, cutoff_fraction, cutoff_order
        )
    if cutoff_fraction is None:
        return field
    if not 0 < cutoff_fraction <= 1:
        raise ValueError("cutoff_fraction must be in (0, 1].")
    n = field.shape[-1]
    k2 = k2_torch(n, lx, ly, field.device, field.dtype)
    grid_nyquist = min(math.pi / (lx / n), math.pi / (ly / n))
    sigma_k = cutoff_fraction * grid_nyquist
    multiplier = torch.exp(-0.5 * k2 / sigma_k**2)
    fh = torch.fft.fft2(field, dim=(-2, -1))
    return torch.fft.ifft2(multiplier * fh, dim=(-2, -1)).real


def recursive_npz_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*.npz") if p.is_file())


def periodic_laplacian_2nd_numpy \
(
    field: np.ndarray, lx: float, ly: float
) -> np.ndarray:
    """Second-order centered periodic Laplacian on the last two dimensions."""
    nx, ny = field.shape[-2:]
    dx = lx / nx
    dy = ly / ny
    dxx = \
    (
        np.roll(field, 1, axis=-2) - 2 * field
        + np.roll(field, -1, axis=-2)
    ) / dx**2
    dyy = \
    (
        np.roll(field, 1, axis=-1) - 2 * field
        + np.roll(field, -1, axis=-1)
    ) / dy**2
    return dxx + dyy


def periodic_laplacian_2nd_torch \
(
    field: torch.Tensor, lx: float, ly: float
) -> torch.Tensor:
    """Differentiable second-order centered periodic Laplacian."""
    nx, ny = field.shape[-2:]
    dx = lx / nx
    dy = ly / ny
    dxx = \
    (
        torch.roll(field, 1, dims=-2) - 2 * field
        + torch.roll(field, -1, dims=-2)
    ) / dx**2
    dyy = \
    (
        torch.roll(field, 1, dims=-1) - 2 * field
        + torch.roll(field, -1, dims=-1)
    ) / dy**2
    return dxx + dyy


def periodic_laplacian_4th_numpy \
(
    field: np.ndarray, lx: float, ly: float
) -> np.ndarray:
    """Fourth-order periodic Laplacian on the last two array dimensions."""
    nx, ny = field.shape[-2:]
    if nx < 5 or ny < 5:
        raise ValueError \
        (
            "The fourth-order Laplacian requires at least 5 points."
        )
    dx = lx / nx
    dy = ly / ny
    dxx = \
    (
        -np.roll(field, 2, axis=-2)
        + 16 * np.roll(field, 1, axis=-2)
        - 30 * field
        + 16 * np.roll(field, -1, axis=-2)
        - np.roll(field, -2, axis=-2)
    ) / (12 * dx**2)
    dyy = \
    (
        -np.roll(field, 2, axis=-1)
        + 16 * np.roll(field, 1, axis=-1)
        - 30 * field
        + 16 * np.roll(field, -1, axis=-1)
        - np.roll(field, -2, axis=-1)
    ) / (12 * dy**2)
    return dxx + dyy


def periodic_laplacian_4th_torch \
(
    field: torch.Tensor, lx: float, ly: float
) -> torch.Tensor:
    """Differentiable fourth-order periodic Laplacian."""
    nx, ny = field.shape[-2:]
    if nx < 5 or ny < 5:
        raise ValueError \
        (
            "The fourth-order Laplacian requires at least 5 points."
        )
    dx = lx / nx
    dy = ly / ny
    dxx = \
    (
        -torch.roll(field, 2, dims=-2)
        + 16 * torch.roll(field, 1, dims=-2)
        - 30 * field
        + 16 * torch.roll(field, -1, dims=-2)
        - torch.roll(field, -2, dims=-2)
    ) / (12 * dx**2)
    dyy = \
    (
        -torch.roll(field, 2, dims=-1)
        + 16 * torch.roll(field, 1, dims=-1)
        - 30 * field
        + 16 * torch.roll(field, -1, dims=-1)
        - torch.roll(field, -2, dims=-1)
    ) / (12 * dy**2)
    return dxx + dyy


def load_dns_snapshot \
(
    path: Path,
    field_names: Sequence[str],
    expected_n: int,
    lx: float,
    ly: float,
    dtype=np.float32,
) -> np.ndarray:
    if tuple(field_names) != HW_FIELD_NAMES:
        raise ValueError(f"HW fields must be ordered as {HW_FIELD_NAMES}.")
    with np.load(path) as src:
        missing = [name for name in field_names if name not in src]
        if missing:
            raise KeyError(f"{path} is missing fields: {missing}")
        arrays = [np.asarray(src[name], dtype=dtype) for name in field_names]
    for name, a in zip(field_names, arrays):
        if a.shape != (expected_n, expected_n):
            raise ValueError(f"{path}: {name} has shape {a.shape}")
        if not np.all(np.isfinite(a)):
            raise ValueError(f"{path}: {name} contains NaN or Inf")
    return np.stack(arrays).astype(dtype, copy=False)


def filter_snapshot \
(
    dns: np.ndarray,
    alpha: float,
    lx: float,
    ly: float,
    coarse_n: int,
    dtype=np.float32,
) -> tuple[np.ndarray, np.ndarray]:
    fine = np.stack \
    (
        [germano_filter(channel, alpha, lx, ly) for channel in dns]
    ).astype(dtype)
    coarse = np.stack \
    (
        [spectral_resize(channel, coarse_n) for channel in fine]
    ).astype(dtype)
    return coarse, fine


def calculate_filtered_dns_shell_power \
(
    files, field_names, fine_n, coarse_n, alpha, lx, ly,
    output_channels, dtype=np.float32,
):
    """Average filtered-DNS Fourier power per channel and radial shell."""
    kx = 2 * np.pi * np.fft.fftfreq(fine_n, d=lx / fine_n)
    ky = 2 * np.pi * np.fft.fftfreq(fine_n, d=ly / fine_n)
    dk = min(2 * np.pi / lx, 2 * np.pi / ly)
    shells = np.rint \
    (
        np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2) / dk
    ).astype(np.int64)
    shell_count = int(shells.max()) + 1
    counts = np.bincount(shells.ravel(), minlength=shell_count)
    total = np.zeros((output_channels, shell_count), dtype=np.float64)
    for index, path in enumerate(files, 1):
        dns = load_dns_snapshot \
        (
            path, field_names, fine_n, lx, ly, dtype=dtype
        )
        _, filtered = filter_snapshot \
        (
            dns, alpha, lx, ly, coarse_n, dtype=dtype
        )
        power = np.abs \
        (
            np.fft.fft2(filtered[:output_channels], axes=(-2, -1))
        ) ** 2
        for channel in range(output_channels):
            total[channel] += np.bincount \
            (
                shells.ravel(), weights=power[channel].ravel(),
                minlength=shell_count,
            )
        if index == 1 or index % 100 == 0 or index == len(files):
            print(f"Filtered-DNS spectrum: {index}/{len(files)}")
    return (total / (len(files) * counts[None, :])).astype(dtype)


def calculate_training_statistics \
(
    files: Sequence[Path],
    field_names: Sequence[str],
    fine_n: int,
    coarse_n: int,
    alpha: float,
    lx: float,
    ly: float,
    dtype=np.float32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = len(field_names)
    s = np.zeros(c)
    ss = np.zeros(c)
    dns_s = np.zeros(c)
    dns_ss = np.zeros(c)
    npts = 0
    for i, p in enumerate(files, 1):
        dns = load_dns_snapshot \
        (
            p, field_names, fine_n, lx, ly, dtype=dtype
        )
        _, fine = filter_snapshot \
        (
            dns, alpha, lx, ly, coarse_n, dtype=dtype
        )
        s += fine.sum((1, 2), dtype=np.float64)
        ss += np.square(fine, dtype=np.float64).sum((1, 2))
        dns_s += dns.sum((1, 2), dtype=np.float64)
        dns_ss += np.square(dns, dtype=np.float64).sum((1, 2))
        npts += fine_n * fine_n
        if i == 1 or i % 100 == 0 or i == len(files):
            print(f"Normalisation statistics: {i}/{len(files)}")
    mean = s / npts
    var = np.maximum(ss / npts - mean**2, 0)
    std = np.maximum(np.sqrt(var), 1e-8)
    dns_mean = dns_s / npts
    dns_var = np.maximum(dns_ss / npts - dns_mean**2, 0)
    dns_std = np.maximum(np.sqrt(dns_var), 1e-8)
    return \
    (
        mean.astype(dtype).reshape(1, c, 1, 1),
        std.astype(dtype).reshape(1, c, 1, 1),
        dns_std.astype(dtype).reshape(1, c, 1, 1),
    )


class GermanoSnapshotDataset(Dataset):
    def __init__ \
    (
        self, files, field_names, fine_n, coarse_n, alpha, lx, ly, mean, std,
        random_periodic_shift=False,
        periodic_directions=(True, True),
    ):
        self.files = list(files)
        self.field_names = tuple(field_names)
        self.fine_n = fine_n
        self.coarse_n = coarse_n
        self.alpha = alpha
        self.lx = lx
        self.ly = ly
        self.mean = mean[0]
        self.std = std[0]
        self.dtype = mean.dtype
        self.periodic_directions = tuple(bool(v) for v in periodic_directions)
        if len(self.periodic_directions) != 2:
            raise ValueError("periodic_directions must contain x and y flags.")
        self.random_periodic_shift = \
        (
            random_periodic_shift and all(self.periodic_directions)
        )
        self.scale_factor = fine_n // coarse_n

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        dns = load_dns_snapshot \
        (
            self.files[index],
            self.field_names,
            self.fine_n,
            self.lx,
            self.ly,
            dtype=self.dtype,
        )
        coarse, fine = filter_snapshot \
        (
            dns,
            self.alpha,
            self.lx,
            self.ly,
            self.coarse_n,
            dtype=self.dtype,
        )
        if self.random_periodic_shift:
            coarse_shift = tuple \
            (
                int(value)
                for value in np.random.randint(0, self.coarse_n, size=2)
            )
            fine_shift = tuple \
            (
                value * self.scale_factor for value in coarse_shift
            )
            coarse = np.roll(coarse, coarse_shift, axis=(-2, -1))
            fine = np.roll(fine, fine_shift, axis=(-2, -1))
            dns = np.roll(dns, fine_shift, axis=(-2, -1))
        return \
        (
            torch.from_numpy((coarse - self.mean) / self.std),
            torch.from_numpy((fine - self.mean) / self.std),
            torch.from_numpy(dns),
        )


class ConvBlock(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        g = min(8, oc)
        while oc % g:
            g -= 1
        self.net = nn.Sequential \
        (
            nn.Conv2d(ic, oc, 3, padding=1, padding_mode="circular"),
            nn.GroupNorm(g, oc),
            nn.GELU(),
            nn.Conv2d(oc, oc, 3, padding=1, padding_mode="circular"),
            nn.GroupNorm(g, oc),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


def periodic_interpolate(x, scale_factor, mode):
    """Upsample with an interpolation stencil that wraps periodically."""
    if mode not in ("bilinear", "bicubic"):
        raise ValueError \
        (
            "Periodic interpolation supports bilinear or bicubic mode."
        )
    if isinstance(scale_factor, int):
        scales = (scale_factor, scale_factor)
    else:
        scales = tuple(scale_factor)
    if len(scales) != 2 or any(scale < 1 for scale in scales):
        raise ValueError("scale_factor must contain two positive integers.")
    if any(int(scale) != scale for scale in scales):
        raise ValueError \
        (
            "Periodic interpolation requires integer scale factors."
        )
    scale_x, scale_y = (int(scale) for scale in scales)
    radius = 1 if mode == "bilinear" else 2
    padded = nn.functional.pad \
    (
        x,
        (radius, radius, radius, radius),
        mode="circular",
    )
    upsampled = nn.functional.interpolate \
    (
        padded,
        scale_factor=(scale_x, scale_y),
        mode=mode,
        align_corners=False,
    )
    crop_x = radius * scale_x
    crop_y = radius * scale_y
    return upsampled[..., crop_x:-crop_x, crop_y:-crop_y]


class PeriodicLowPass2d(nn.Module):
    """Fixed separable [1, 3, 3, 1] FIR filter with unit DC gain."""

    def __init__(self):
        super().__init__()
        taps = torch.tensor([1., 3., 3., 1.])
        kernel = (taps[:, None] * taps[None, :]) / taps.sum().square()
        self.register_buffer("kernel", kernel[None, None], persistent=False)

    def forward(self, x):
        # The even-length stencil needs asymmetric padding to retain size.
        x = nn.functional.pad(x, (1, 2, 1, 2), mode="circular")
        return nn.functional.conv2d \
        (
            x,
            self.kernel.to(dtype=x.dtype).expand(x.shape[1], 1, 4, 4),
            groups=x.shape[1],
        )


class UpBlock(nn.Module):
    def __init__(self, ic, sc, oc):
        super().__init__()
        self.up = nn.ConvTranspose2d(ic, ic, 4, stride=2, padding=1)
        self.blur = PeriodicLowPass2d()
        self.net = ConvBlock(ic + sc, oc)

    def forward(self, x, skip):
        if any(s != 2 * n for n, s in zip(x.shape[-2:], skip.shape[-2:])):
            raise ValueError \
            (
                "Skip dimensions must be twice the input dimensions."
            )
        # Supply periodic neighbors before convolution, then remove their
        # doubled halo so the retained output has no zero-padded boundary.
        x = nn.functional.pad(x, (1, 1, 1, 1), mode="circular")
        x = self.up(x)[..., 2:-2, 2:-2]
        x = self.blur(x)
        return self.net(torch.cat((x, skip), 1))


class ResidualUNetSR(nn.Module):
    def __init__ \
    (
        self, channels=3, base_channels=24, scale_factor=8, output_channels=2,
        variational=True
    ):
        super().__init__()
        b = base_channels
        self.scale_factor = scale_factor
        self.pool = nn.AvgPool2d(2)
        self.e1 = ConvBlock(channels, b)
        self.e2 = ConvBlock(b, 2 * b)
        self.e3 = ConvBlock(2 * b, 4 * b)
        self.e4 = ConvBlock(4 * b, 8 * b)
        self.bn = ConvBlock(8 * b, 8 * b)
        self.variational = variational
        if variational:
            self.latent_mu = nn.Conv2d(8 * b, 8 * b, 1)
            self.latent_logvar = nn.Conv2d(8 * b, 8 * b, 1)
        self.d3 = UpBlock(8 * b, 4 * b, 4 * b)
        self.d2 = UpBlock(4 * b, 2 * b, 2 * b)
        self.d1 = UpBlock(2 * b, b, b)
        self.output_channels = output_channels
        self.out = nn.Conv2d(b, output_channels, 1)

    def forward(self, coarse, *, sample=None, return_kl=False):
        """Sample during training; use the posterior mean in evaluation.

        Set sample=True for stochastic inference. KL is averaged over the
        batch, latent channels and spatial cells, against a unit Gaussian.
        """
        base = periodic_interpolate \
        (
            coarse,
            self.scale_factor,
            "bicubic",
        )
        e1 = self.e1(base)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        bn = self.bn(e4)
        kl = bn.new_zeros(())
        if self.variational:
            mu = self.latent_mu(bn)
            logvar = self.latent_logvar(bn).clamp(-20.0, 10.0)
            kl = 0.5 * (mu.square() + logvar.exp() - 1.0 - logvar).mean()
            sample = self.training if sample is None else sample
            bn = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if sample else mu
        correction = self.out(self.d1(self.d2(self.d3(bn, e3), e2), e1))
        prediction = base[:, : self.output_channels] + correction
        return (prediction, kl) if return_kl else prediction



class PeriodicResidualBlock(nn.Module):
    """Residual convolutions on a fixed periodic grid."""

    def __init__(self, channels, dilation):
        super().__init__()
        groups = min(8, channels)
        while channels % groups:
            groups -= 1
        self.net = nn.Sequential \
        (
            nn.Conv2d \
            (
                channels, channels, 3, padding=dilation,
                dilation=dilation, padding_mode="circular",
            ),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d \
            (
                channels, channels, 3, padding=dilation,
                dilation=dilation, padding_mode="circular",
            ),
            nn.GroupNorm(groups, channels),
        )
        self.activation = nn.GELU()

    def forward(self, x):
        return self.activation(x + self.net(x))


class PeriodicResidualCNN(nn.Module):
    """Super-resolution with a constant-resolution periodic residual trunk.

    Only the initial interpolation changes spatial resolution. Evaluation
    uses the latent mean, preserving integer coarse-grid shift equivariance.
    """

    def __init__ \
    (
        self, channels=3, base_channels=24, scale_factor=8, output_channels=2,
        variational=True,
    ):
        super().__init__()
        self.scale_factor = scale_factor
        self.output_channels = output_channels
        self.variational = variational
        self.stem = ConvBlock(channels, base_channels)
        self.blocks = nn.Sequential \
        (
            *[PeriodicResidualBlock(base_channels, d) for d in (1, 2, 4, 1, 2, 4)]
        )
        if variational:
            self.latent_mu = nn.Conv2d(base_channels, base_channels, 1)
            self.latent_logvar = nn.Conv2d(base_channels, base_channels, 1)
        self.out = nn.Conv2d(base_channels, output_channels, 1)

    def forward(self, coarse, *, sample=None, return_kl=False):
        base = periodic_interpolate(coarse, self.scale_factor, "bicubic")
        features = self.blocks(self.stem(base))
        kl = features.new_zeros(())
        if self.variational:
            mu = self.latent_mu(features)
            logvar = self.latent_logvar(features).clamp(-20.0, 10.0)
            kl = 0.5 * (mu.square() + logvar.exp() - 1.0 - logvar).mean()
            sample = self.training if sample is None else sample
            features = \
            (
                mu + torch.randn_like(mu) * (0.5 * logvar).exp()
                if sample else mu
            )
        prediction = base[:, :self.output_channels] + self.out(features)
        return (prediction, kl) if return_kl else prediction



class PeriodicSubpixelCNN(PeriodicResidualCNN):
    """Periodic coarse-grid residual trunk with learned subpixel upsampling.

    Predict complete fine fields, without interpolation or a global input skip.
    The parent architecture remains available for older checkpoints.
    """

    def __init__ \
    (
        self, channels=3, base_channels=24, scale_factor=8, output_channels=2,
        variational=True,
    ):
        if not isinstance(scale_factor, int) or scale_factor < 1:
            raise ValueError("scale_factor must be a positive integer.")
        super().__init__ \
        (
            channels, base_channels, scale_factor, output_channels, variational
        )
        self.out = nn.Conv2d \
        (
            base_channels, output_channels * scale_factor**2, 3,
            padding=1, padding_mode="circular",
        )
        self.shuffle = nn.PixelShuffle(scale_factor)

    def forward(self, coarse, *, sample=None, return_kl=False):
        features = self.blocks(self.stem(coarse))
        kl = features.new_zeros(())
        if self.variational:
            mu = self.latent_mu(features)
            logvar = self.latent_logvar(features).clamp(-20.0, 10.0)
            kl = 0.5 * (mu.square() + logvar.exp() - 1.0 - logvar).mean()
            sample = self.training if sample is None else sample
            features = \
            (
                mu + torch.randn_like(mu) * (0.5 * logvar).exp()
                if sample else mu
            )
        prediction = self.shuffle(self.out(features))
        return (prediction, kl) if return_kl else prediction


def checkpoint_model_class(checkpoint):
    """Dispatch explicitly; checkpoints without metadata use the legacy U-Net."""
    architecture = checkpoint.get("architecture", "ResidualUNetSR")
    if architecture == "PeriodicSubpixelCNN":
        return PeriodicSubpixelCNN
    if architecture == "PeriodicResidualCNN":
        return PeriodicResidualCNN
    if architecture == "ResidualUNetSR":
        return ResidualUNetSR
    raise ValueError(f"Unknown checkpoint architecture: {architecture!r}")


def calculate_losses \
(
    model,
    coarse,
    filtered_target_normalised,
    dns_target,
    mean,
    std,
    dns_std,
    alpha,
    lx,
    ly,
    lambda_dns,
    vorticity_from_potential,
    inverse_cutoff_fraction,
    inverse_cutoff_order,
    post_cutoff_fraction=None,
    post_cutoff_order=8,
    post_filter_type="high_order",
    inverse_method="exact",
    van_cittert_iterations=1,
    van_cittert_relaxation=1.0,
    inverse_filter_type="cutoff",
    filtered_dns_shell_power=None,
    inverse_min_wavenumber=1.0,
):
    output_channels = 2 if vorticity_from_potential else 3
    output_mean = mean[:, :output_channels]
    output_std = std[:, :output_channels]
    prediction, kl = model(coarse, return_kl=True)
    pf = prediction * output_std + output_mean
    pn = (pf - output_mean) / output_std
    inverse_input = inverse_input_filter_torch \
    (
        pf,
        lx,
        ly,
        inverse_filter_type,
        inverse_cutoff_fraction,
        inverse_cutoff_order,
        filtered_dns_shell_power,
        min_wavenumber=inverse_min_wavenumber,
    )
    pd = reconstruct_germano_torch \
    (
        inverse_input,
        alpha,
        lx,
        ly,
        inverse_method,
        van_cittert_iterations,
        van_cittert_relaxation,
    )
    pd = post_filter_torch \
    (
        pd,
        lx,
        ly,
        post_filter_type,
        post_cutoff_fraction,
        post_cutoff_order,
    )
    if vorticity_from_potential:
        vorticity = periodic_laplacian_2nd_torch(pd[:, 1:2], lx, ly)
        pd = torch.cat((pd, vorticity), dim=1)
    # pd = enforce_zero_spatial_mean(pd)
    # pd = restore_channel_spatial_mean(pd, dns_target, 2)
    lf = nn.functional.l1_loss \
    (
        pn, filtered_target_normalised[:, :output_channels]
    )
    dns_absolute_error = torch.abs((pd - dns_target) / dns_std)
    ld = dns_absolute_error.mean()
    total = lf + lambda_dns * ld
    return \
    (
        total,
        {
            "total": total,
            "filtered": lf,
            "kl": kl,
            "dns": ld,
            "dns_U": dns_absolute_error[:, 0].mean(),
            "dns_V": dns_absolute_error[:, 1].mean(),
            "dns_P": dns_absolute_error[:, 2].mean(),
        },
        pf,
        pd,
    )


def hw_energy_spectrum_torch(fields, lx, ly):
    """Return HW energy per Fourier mode from physical n and phi fields.

    E_k = 1/2 (n_k n_k* + (d_x phi)_k (d_x phi)_k*
                         + (d_y phi)_k (d_y phi)_k*)
    """
    if fields.shape[-3] < 2:
        raise ValueError("HW energy requires density and potential channels.")
    nx, ny = fields.shape[-2:]
    density_hat = torch.fft.fft2(fields[..., 0, :, :], dim=(-2, -1))
    potential_hat = torch.fft.fft2(fields[..., 1, :, :], dim=(-2, -1))
    kx = 2 * math.pi * torch.fft.fftfreq \
    (
        nx, d=lx / nx, device=fields.device, dtype=fields.dtype
    )
    ky = 2 * math.pi * torch.fft.fftfreq \
    (
        ny, d=ly / ny, device=fields.device, dtype=fields.dtype
    )
    dphi_dx_hat = 1j * kx[:, None] * potential_hat
    dphi_dy_hat = 1j * ky[None, :] * potential_hat
    normalization = float(nx * ny) ** 2
    return 0.5 * \
    (
        density_hat.abs().square()
        + dphi_dx_hat.abs().square()
        + dphi_dy_hat.abs().square()
    ) / normalization


def above_les_cutoff_mask \
(
    fine_shape, coarse_shape, lx, ly, device=None, dtype=None
):
    """Select Fourier modes whose radial wavenumber exceeds LES Nyquist."""
    fine_nx, fine_ny = fine_shape
    coarse_nx, coarse_ny = coarse_shape
    dtype = dtype or torch.get_default_dtype()
    kx = 2 * math.pi * torch.fft.fftfreq \
    (
        fine_nx, d=lx / fine_nx, device=device, dtype=dtype
    )
    ky = 2 * math.pi * torch.fft.fftfreq \
    (
        fine_ny, d=ly / fine_ny, device=device, dtype=dtype
    )
    les_cutoff = min \
    (
        math.pi * coarse_nx / lx,
        math.pi * coarse_ny / ly,
    )
    return torch.sqrt(kx[:, None].square() + ky[None, :].square()) > les_cutoff


def relative_l2(reference, prediction):
    num = torch.linalg.vector_norm((prediction - reference).flatten(1), dim=1)
    den = torch.linalg.vector_norm(reference.flatten(1), dim=1).clamp_min(1e-12)
    return (num / den).mean()


def smooth_spectrum(values, window=5, preserve_bins=4):
    """Return a moving-average spectrum while preserving its leading bins."""
    values = np.asarray(values)
    if values.ndim != 1:
        raise ValueError("Spectrum values must be one-dimensional.")
    if window <= 0:
        raise ValueError("Spectrum smoothing window must be positive.")
    if preserve_bins < 0:
        raise ValueError("The number of preserved spectrum bins cannot be negative.")
    if values.size < window:
        return values.copy()
    smoothed = np.convolve(values, np.ones(window) / window, mode="same")
    smoothed[:preserve_bins] = values[:preserve_bins]
    return smoothed


def isotropic_spectrum(field, lx, ly, smooth=False):
    n = field.shape[0]
    kx = 2 * np.pi * np.fft.fftfreq(n, d=lx / n)
    ky = 2 * np.pi * np.fft.fftfreq(n, d=ly / n)
    kxg, kyg = np.meshgrid(kx, ky, indexing="ij")
    mag = np.sqrt(kxg**2 + kyg**2)
    power = np.abs(np.fft.fft2(field)) ** 2 / n**4
    dk = min(2 * np.pi / lx, 2 * np.pi / ly)
    shell = np.rint(mag / dk).astype(int)
    p = np.bincount(shell.ravel(), weights=power.ravel())
    c = np.bincount(shell.ravel())
    k = np.arange(p.size) * dk
    v = c > 0
    p = p[v]
    if smooth:
        p = smooth_spectrum(p)
    return k[v], p


def isotropic_hw_energy_spectrum(fields, lx, ly, smooth=False):
    """Return the shell-summed HW energy spectrum from physical n and phi.

    E_k = 1/2 (n_k n_k* + (d_x phi)_k (d_x phi)_k*
                         + (d_y phi)_k (d_y phi)_k*)
    """
    if fields.ndim != 3 or fields.shape[0] < 2:
        raise ValueError \
        (
            "HW energy requires fields shaped (channels, nx, ny)."
        )
    nx, ny = fields.shape[-2:]
    density_hat = np.fft.fft2(fields[0])
    potential_hat = np.fft.fft2(fields[1])
    kx = 2 * np.pi * np.fft.fftfreq(nx, d=lx / nx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=ly / ny)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    dphi_dx_hat = 1j * kx_grid * potential_hat
    dphi_dy_hat = 1j * ky_grid * potential_hat
    energy = 0.5 * np.real \
    (
        np.conj(density_hat) * density_hat
        + np.conj(dphi_dx_hat) * dphi_dx_hat
        + np.conj(dphi_dy_hat) * dphi_dy_hat
    ) / float(nx * ny) ** 2
    wavenumber = np.sqrt(kx_grid**2 + ky_grid**2)
    dk = min(2 * np.pi / lx, 2 * np.pi / ly)
    shell = np.rint(wavenumber / dk).astype(int)
    shell_energy = np.bincount(shell.ravel(), weights=energy.ravel())
    k = np.arange(shell_energy.size) * dk
    if smooth:
        shell_energy = smooth_spectrum(shell_energy)
    return k, shell_energy


def save_image(field, title, path, cmap=None, symmetric=False):
    nx, ny = field.shape
    fig, ax = plt.subplots(figsize=(6.3, 5.2))
    limits = {}
    if symmetric:
        maximum = float(np.max(np.abs(field)))
        if maximum > 0:
            limits = {"vmin": -maximum, "vmax": maximum}
    im = ax.imshow \
    (
        field.T,
        origin="lower",
        extent=(0, nx, 0, ny),
        aspect="equal",
        cmap=cmap,
        **limits,
    )
    ax.set_title(title)
    ax.set_xlabel("x [pixels]")
    ax.set_ylabel("y [pixels]")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_spectrum_plot \
(
    dns,
    filtered_fine,
    filtered_coarse,
    predicted_filtered,
    predicted_dns,
    coarse_n,
    lx,
    ly,
    path,
    inverse_input_label="U-Net filtered",
    smooth_spectrum=False,
):
    spectra = \
    {
        "DNS": isotropic_hw_energy_spectrum \
        (
            dns, lx, ly, smooth=smooth_spectrum
        ),
        f"Filtered {filtered_fine.shape[-1]}": isotropic_hw_energy_spectrum \
        (
            filtered_fine, lx, ly, smooth=smooth_spectrum
        ),
        f"Filtered {coarse_n}": isotropic_hw_energy_spectrum \
        (
            filtered_coarse, lx, ly, smooth=smooth_spectrum
        ),
        inverse_input_label: isotropic_hw_energy_spectrum \
        (
            predicted_filtered, lx, ly, smooth=smooth_spectrum
        ),
        "Reconstructed DNS": isotropic_hw_energy_spectrum \
        (
            predicted_dns, lx, ly, smooth=smooth_spectrum
        ),
    }
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    for label, (k, p) in spectra.items():
        v = (k > 0) & (p > 0)
        if label == f"Filtered {coarse_n}":
            ax.loglog \
            (
                k[v],
                p[v],
                linestyle="None",
                marker="o",
                markersize=4,
                label=label,
            )
        elif label == inverse_input_label:
            ax.loglog(k[v], p[v], linestyle="--", linewidth=1.8, label=label)
        else:
            ax.loglog(k[v], p[v], linewidth=1.8, label=label)
    kn = min(np.pi / (lx / coarse_n), np.pi / (ly / coarse_n))
    ax.axvline \
    (
        kn, linestyle=":", linewidth=1.4, label=f"{coarse_n}-grid Nyquist"
    )
    ax.set_title \
    (
        r"HW energy spectrum: $E=\frac{1}{2}\int(n^2+|\nabla\phi|^2)$"
    )
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"Shell energy $E(k)$")
    ax.set_ylim(bottom=SPECTRUM_Y_MIN, top=SPECTRUM_Y_MAX)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(numpoints=1)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
