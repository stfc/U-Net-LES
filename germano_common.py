#!/usr/bin/env python3
"""Common components for Germano-filter U-Net training and inference."""

from __future__ import annotations
import math, random
from pathlib import Path
from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

HW_FIELD_NAMES = ("U", "V", "W")
HW_FIELD_LABELS = {
    "U": r"electron density $n_e$",
    "V": r"electrostatic potential $\phi$",
    "W": r"vorticity $\omega = \nabla^2\phi$",
}


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
        raise ValueError(
            "This checkpoint predates the Hasegawa-Wakatani constraints. "
            "Retrain it with fields U, V, W before inference or resuming."
        )
    mode = checkpoint.get("vorticity_mode", "potential_laplacian")
    if mode not in ("potential_laplacian", "learned_channel"):
        raise ValueError(f"Unsupported checkpoint vorticity mode: {mode}")
    operator = checkpoint.get(
        "vorticity_operator",
        checkpoint.get("training_parameters", {}).get(
            "vorticity_operator", "fourth_order_periodic_laplacian"
        ),
    )
    supported = {
        "learned_independent_channel",
        "second_order_periodic_laplacian",
        "fourth_order_periodic_laplacian",
    }
    if operator not in supported:
        raise ValueError(f"Unsupported checkpoint vorticity operator: {operator}")


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


def germano_filter(
    field: np.ndarray, alpha: float, lx: float, ly: float
) -> np.ndarray:
    fh = np.fft.fft2(field)
    return np.fft.ifft2(
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
    th[ts : ts + copy_n, ts : ts + copy_n] = sh[
        ss : ss + copy_n, ss : ss + copy_n
    ]
    th *= (target_n / source_n) ** 2
    return np.fft.ifft2(np.fft.ifftshift(th)).real


def k2_torch(
    n: int, lx: float, ly: float, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    kx = (
        2 * math.pi * torch.fft.fftfreq(n, d=lx / n, device=device, dtype=dtype)
    )
    ky = (
        2 * math.pi * torch.fft.fftfreq(n, d=ly / n, device=device, dtype=dtype)
    )
    kxg, kyg = torch.meshgrid(kx, ky, indexing="ij")
    return kxg.square() + kyg.square()


def inverse_germano_torch(
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
        mult = mult * torch.exp(
            -(torch.sqrt(k2) / cutoff_wavenumber) ** cutoff_order
        )
    return torch.fft.ifft2(mult * fh, dim=(-2, -1)).real


def spectral_cutoff_torch(
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
    multiplier = torch.exp(
        -(torch.sqrt(k2) / cutoff_wavenumber) ** cutoff_order
    )
    fh = torch.fft.fft2(field, dim=(-2, -1))
    return torch.fft.ifft2(multiplier * fh, dim=(-2, -1)).real


def recursive_npz_files(directory: Path) -> list[Path]:
    return sorted(p for p in directory.rglob("*.npz") if p.is_file())


def periodic_laplacian_2nd_numpy(
    field: np.ndarray, lx: float, ly: float
) -> np.ndarray:
    """Second-order centered periodic Laplacian on the last two dimensions."""
    nx, ny = field.shape[-2:]
    dx = lx / nx
    dy = ly / ny
    dxx = (
        np.roll(field, 1, axis=-2) - 2 * field
        + np.roll(field, -1, axis=-2)
    ) / dx**2
    dyy = (
        np.roll(field, 1, axis=-1) - 2 * field
        + np.roll(field, -1, axis=-1)
    ) / dy**2
    return dxx + dyy


def periodic_laplacian_2nd_torch(
    field: torch.Tensor, lx: float, ly: float
) -> torch.Tensor:
    """Differentiable second-order centered periodic Laplacian."""
    nx, ny = field.shape[-2:]
    dx = lx / nx
    dy = ly / ny
    dxx = (
        torch.roll(field, 1, dims=-2) - 2 * field
        + torch.roll(field, -1, dims=-2)
    ) / dx**2
    dyy = (
        torch.roll(field, 1, dims=-1) - 2 * field
        + torch.roll(field, -1, dims=-1)
    ) / dy**2
    return dxx + dyy


def periodic_laplacian_4th_numpy(
    field: np.ndarray, lx: float, ly: float
) -> np.ndarray:
    """Fourth-order periodic Laplacian on the last two array dimensions."""
    nx, ny = field.shape[-2:]
    if nx < 5 or ny < 5:
        raise ValueError(
            "The fourth-order Laplacian requires at least 5 points."
        )
    dx = lx / nx
    dy = ly / ny
    dxx = (
        -np.roll(field, 2, axis=-2)
        + 16 * np.roll(field, 1, axis=-2)
        - 30 * field
        + 16 * np.roll(field, -1, axis=-2)
        - np.roll(field, -2, axis=-2)
    ) / (12 * dx**2)
    dyy = (
        -np.roll(field, 2, axis=-1)
        + 16 * np.roll(field, 1, axis=-1)
        - 30 * field
        + 16 * np.roll(field, -1, axis=-1)
        - np.roll(field, -2, axis=-1)
    ) / (12 * dy**2)
    return dxx + dyy


def periodic_laplacian_4th_torch(
    field: torch.Tensor, lx: float, ly: float
) -> torch.Tensor:
    """Differentiable fourth-order periodic Laplacian."""
    nx, ny = field.shape[-2:]
    if nx < 5 or ny < 5:
        raise ValueError(
            "The fourth-order Laplacian requires at least 5 points."
        )
    dx = lx / nx
    dy = ly / ny
    dxx = (
        -torch.roll(field, 2, dims=-2)
        + 16 * torch.roll(field, 1, dims=-2)
        - 30 * field
        + 16 * torch.roll(field, -1, dims=-2)
        - torch.roll(field, -2, dims=-2)
    ) / (12 * dx**2)
    dyy = (
        -torch.roll(field, 2, dims=-1)
        + 16 * torch.roll(field, 1, dims=-1)
        - 30 * field
        + 16 * torch.roll(field, -1, dims=-1)
        - torch.roll(field, -2, dims=-1)
    ) / (12 * dy**2)
    return dxx + dyy


def enforce_vorticity_from_potential_torch(
    fields: torch.Tensor, lx: float, ly: float, laplacian_order: int = 2
) -> torch.Tensor:
    """Replace W with the selected periodic Laplacian of potential V."""
    if fields.shape[-3] < 2:
        raise ValueError("Vorticity enforcement requires U and V channels.")
    operators = {
        2: periodic_laplacian_2nd_torch,
        4: periodic_laplacian_4th_torch,
    }
    if laplacian_order not in operators:
        raise ValueError("laplacian_order must be 2 or 4.")
    vorticity = operators[laplacian_order](
        fields[..., 1:2, :, :], lx, ly
    )
    return torch.cat((fields[..., :2, :, :], vorticity), dim=-3)


def hw_fields_numpy(
    density: np.ndarray,
    potential: np.ndarray,
    lx: float,
    ly: float,
    dtype=np.float32,
    laplacian_order: int = 2,
) -> np.ndarray:
    """Apply HW constraints and derive vorticity from potential."""
    density = density - density.mean(axis=(-2, -1), keepdims=True)
    potential = potential - potential.mean(axis=(-2, -1), keepdims=True)
    operators = {
        2: periodic_laplacian_2nd_numpy,
        4: periodic_laplacian_4th_numpy,
    }
    if laplacian_order not in operators:
        raise ValueError("laplacian_order must be 2 or 4.")
    vorticity = operators[laplacian_order](potential, lx, ly)
    return np.stack((density, potential, vorticity), axis=-3).astype(dtype)


def hw_fields_torch(
    density: torch.Tensor,
    potential: torch.Tensor,
    lx: float,
    ly: float,
    laplacian_order: int = 2,
) -> torch.Tensor:
    """Differentiably apply HW constraints and derive vorticity."""
    density = density - density.mean(dim=(-2, -1), keepdim=True)
    potential = potential - potential.mean(dim=(-2, -1), keepdim=True)
    operators = {
        2: periodic_laplacian_2nd_torch,
        4: periodic_laplacian_4th_torch,
    }
    if laplacian_order not in operators:
        raise ValueError("laplacian_order must be 2 or 4.")
    vorticity = operators[laplacian_order](potential, lx, ly)
    return torch.cat((density, potential, vorticity), dim=-3)


def load_dns_snapshot(
    path: Path,
    field_names: Sequence[str],
    expected_n: int,
    lx: float,
    ly: float,
    dtype=np.float32,
    laplacian_order: int = 2,
) -> np.ndarray:
    if tuple(field_names) != HW_FIELD_NAMES:
        raise ValueError(f"HW fields must be ordered as {HW_FIELD_NAMES}.")
    with np.load(path) as src:
        missing = [name for name in ("U", "V") if name not in src]
        if missing:
            raise KeyError(f"{path} is missing fields: {missing}")
        arrays = [np.asarray(src[name], dtype=dtype) for name in ("U", "V")]
    for name, a in zip(("U", "V"), arrays):
        if a.shape != (expected_n, expected_n):
            raise ValueError(f"{path}: {name} has shape {a.shape}")
        if not np.all(np.isfinite(a)):
            raise ValueError(f"{path}: {name} contains NaN or Inf")
    return hw_fields_numpy(
        arrays[0], arrays[1], lx, ly,
        dtype=dtype, laplacian_order=laplacian_order
    )


def filter_snapshot(
    dns: np.ndarray,
    alpha: float,
    lx: float,
    ly: float,
    coarse_n: int,
    dtype=np.float32,
    laplacian_order: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    filtered_density = germano_filter(dns[0], alpha, lx, ly)
    filtered_potential = germano_filter(dns[1], alpha, lx, ly)
    fine = hw_fields_numpy(
        filtered_density, filtered_potential, lx, ly, dtype=dtype,
        laplacian_order=laplacian_order
    )
    coarse_density = spectral_resize(fine[0], coarse_n)
    coarse_potential = spectral_resize(fine[1], coarse_n)
    coarse = hw_fields_numpy(
        coarse_density, coarse_potential, lx, ly, dtype=dtype,
        laplacian_order=laplacian_order
    )
    return coarse, fine


def calculate_training_statistics(
    files: Sequence[Path],
    field_names: Sequence[str],
    fine_n: int,
    coarse_n: int,
    alpha: float,
    lx: float,
    ly: float,
    dtype=np.float32,
    laplacian_order: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    c = len(field_names)
    s = np.zeros(c)
    ss = np.zeros(c)
    dns_s = np.zeros(c)
    dns_ss = np.zeros(c)
    npts = 0
    for i, p in enumerate(files, 1):
        dns = load_dns_snapshot(
            p, field_names, fine_n, lx, ly, dtype=dtype,
            laplacian_order=laplacian_order
        )
        _, fine = filter_snapshot(
            dns, alpha, lx, ly, coarse_n, dtype=dtype,
            laplacian_order=laplacian_order
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
    return (
        mean.astype(dtype).reshape(1, c, 1, 1),
        std.astype(dtype).reshape(1, c, 1, 1),
        dns_std.astype(dtype).reshape(1, c, 1, 1),
    )


class GermanoSnapshotDataset(Dataset):
    def __init__(
        self, files, field_names, fine_n, coarse_n, alpha, lx, ly, mean, std,
        laplacian_order=2, random_periodic_shift=False,
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
        self.laplacian_order = laplacian_order
        self.periodic_directions = tuple(bool(v) for v in periodic_directions)
        if len(self.periodic_directions) != 2:
            raise ValueError("periodic_directions must contain x and y flags.")
        self.random_periodic_shift = (
            random_periodic_shift and all(self.periodic_directions)
        )
        self.scale_factor = fine_n // coarse_n

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        dns = load_dns_snapshot(
            self.files[index],
            self.field_names,
            self.fine_n,
            self.lx,
            self.ly,
            dtype=self.dtype, laplacian_order=self.laplacian_order,
        )
        coarse, fine = filter_snapshot(
            dns,
            self.alpha,
            self.lx,
            self.ly,
            self.coarse_n,
            dtype=self.dtype, laplacian_order=self.laplacian_order,
        )
        if self.random_periodic_shift:
            coarse_shift = tuple(
                int(value)
                for value in np.random.randint(0, self.coarse_n, size=2)
            )
            fine_shift = tuple(
                value * self.scale_factor for value in coarse_shift
            )
            coarse = np.roll(coarse, coarse_shift, axis=(-2, -1))
            fine = np.roll(fine, fine_shift, axis=(-2, -1))
            dns = np.roll(dns, fine_shift, axis=(-2, -1))
        return (
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
        self.net = nn.Sequential(
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
        raise ValueError(
            "Periodic interpolation supports bilinear or bicubic mode."
        )
    if isinstance(scale_factor, int):
        scales = (scale_factor, scale_factor)
    else:
        scales = tuple(scale_factor)
    if len(scales) != 2 or any(scale < 1 for scale in scales):
        raise ValueError("scale_factor must contain two positive integers.")
    if any(int(scale) != scale for scale in scales):
        raise ValueError(
            "Periodic interpolation requires integer scale factors."
        )
    scale_x, scale_y = (int(scale) for scale in scales)
    radius = 1 if mode == "bilinear" else 2
    padded = nn.functional.pad(
        x,
        (radius, radius, radius, radius),
        mode="circular",
    )
    upsampled = nn.functional.interpolate(
        padded,
        scale_factor=(scale_x, scale_y),
        mode=mode,
        align_corners=False,
    )
    crop_x = radius * scale_x
    crop_y = radius * scale_y
    return upsampled[..., crop_x:-crop_x, crop_y:-crop_y]


class UpBlock(nn.Module):
    def __init__(self, ic, sc, oc):
        super().__init__()
        self.net = ConvBlock(ic + sc, oc)

    def forward(self, x, skip):
        scale_x = skip.shape[-2] // x.shape[-2]
        scale_y = skip.shape[-1] // x.shape[-1]
        if x.shape[-2] * scale_x != skip.shape[-2]:
            raise ValueError(
                "Skip height must be an integer multiple of input."
            )
        if x.shape[-1] * scale_y != skip.shape[-1]:
            raise ValueError("Skip width must be an integer multiple of input.")
        x = periodic_interpolate(x, (scale_x, scale_y), "bilinear")
        return self.net(torch.cat((x, skip), 1))


class ResidualUNetSR(nn.Module):
    def __init__(
        self, channels=3, base_channels=24, scale_factor=8, output_channels=2
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
        self.d3 = UpBlock(8 * b, 4 * b, 4 * b)
        self.d2 = UpBlock(4 * b, 2 * b, 2 * b)
        self.d1 = UpBlock(2 * b, b, b)
        self.output_channels = output_channels
        self.out = nn.Conv2d(b, output_channels, 1)

    def forward(self, coarse):
        base = periodic_interpolate(
            coarse,
            self.scale_factor,
            "bicubic",
        )
        e1 = self.e1(base)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        e4 = self.e4(self.pool(e3))
        bn = self.bn(e4)
        correction = self.out(self.d1(self.d2(self.d3(bn, e3), e2), e1))
        return base[:, : self.output_channels] + correction


def calculate_losses(
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
    lambda_spectrum,
    lambda_laplacian,
    inverse_cutoff_fraction,
    inverse_cutoff_order,
    post_cutoff_fraction=0.5,
    post_cutoff_order=8,
    learn_vorticity=False,
    laplacian_order=2,
    invert_derived_vorticity=False,
):
    if learn_vorticity and invert_derived_vorticity:
        raise ValueError(
            "invert_derived_vorticity requires learn_vorticity=False"
        )
    if learn_vorticity:
        pf = model(coarse) * std + mean
    else:
        predicted_uv = model(coarse) * std[:, :2] + mean[:, :2]
        pf = hw_fields_torch(
            predicted_uv[:, :1], predicted_uv[:, 1:2], lx, ly,
            laplacian_order=laplacian_order
        )
    pn = (pf - mean) / std
    inverse_input = spectral_cutoff_torch(
        pf,
        lx,
        ly,
        inverse_cutoff_fraction,
        inverse_cutoff_order,
    )
    pd_unconstrained = inverse_germano_torch(
        inverse_input,
        alpha,
        lx,
        ly,
    )
    if learn_vorticity or invert_derived_vorticity:
        pd = pd_unconstrained
    else:
        pd = hw_fields_torch(
            pd_unconstrained[:, :1], pd_unconstrained[:, 1:2], lx, ly,
            laplacian_order=laplacian_order
        )
    pd = spectral_cutoff_torch(
        pd,
        lx,
        ly,
        post_cutoff_fraction,
        post_cutoff_order,
    )
    if not learn_vorticity and not invert_derived_vorticity:
        pd = enforce_vorticity_from_potential_torch(
            pd, lx, ly, laplacian_order
        )
    lf = nn.functional.l1_loss(pn, filtered_target_normalised)
    dns_absolute_error = torch.abs((pd - dns_target) / dns_std)
    ld = dns_absolute_error.mean()
    filtered_target = filtered_target_normalised * std + mean
    predicted_energy = hw_energy_spectrum_torch(pf, lx, ly)
    target_energy = hw_energy_spectrum_torch(filtered_target, lx, ly)
    ls = nn.functional.l1_loss(
        torch.log1p(predicted_energy), torch.log1p(target_energy)
    )
    laplacian = {
        2: periodic_laplacian_2nd_torch,
        4: periodic_laplacian_4th_torch,
    }[laplacian_order]
    reconstructed_laplacian = laplacian(pd[:, 1:2], lx, ly)
    ll = relative_l2(pd[:, 2:3], reconstructed_laplacian)
    spectrum_relative_l2 = relative_l2(target_energy, predicted_energy)
    total = (
        lf
        + lambda_dns * ld
        + lambda_spectrum * ls
        + lambda_laplacian * ll
    )
    return (
        total,
        {
            "total": total,
            "filtered": lf,
            "dns": ld,
            "spectrum": ls,
            "laplacian": ll,
            "laplacian_relative_l2": ll,
            "spectrum_relative_l2": spectrum_relative_l2,
            "dns_U": dns_absolute_error[:, 0].mean(),
            "dns_V": dns_absolute_error[:, 1].mean(),
            "dns_W": dns_absolute_error[:, 2].mean(),
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
    kx = 2 * math.pi * torch.fft.fftfreq(
        nx, d=lx / nx, device=fields.device, dtype=fields.dtype
    )
    ky = 2 * math.pi * torch.fft.fftfreq(
        ny, d=ly / ny, device=fields.device, dtype=fields.dtype
    )
    dphi_dx_hat = 1j * kx[:, None] * potential_hat
    dphi_dy_hat = 1j * ky[None, :] * potential_hat
    normalization = float(nx * ny) ** 2
    return 0.5 * (
        density_hat.abs().square()
        + dphi_dx_hat.abs().square()
        + dphi_dy_hat.abs().square()
    ) / normalization


def relative_l2(reference, prediction):
    num = torch.linalg.vector_norm((prediction - reference).flatten(1), dim=1)
    den = torch.linalg.vector_norm(reference.flatten(1), dim=1).clamp_min(1e-12)
    return (num / den).mean()


def isotropic_spectrum(field, lx, ly):
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
    return k[v], p[v]


def isotropic_hw_energy_spectrum(fields, lx, ly):
    """Return the shell-summed HW energy spectrum from physical n and phi.

    E_k = 1/2 (n_k n_k* + (d_x phi)_k (d_x phi)_k*
                         + (d_y phi)_k (d_y phi)_k*)
    """
    if fields.ndim != 3 or fields.shape[0] < 2:
        raise ValueError(
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
    energy = 0.5 * np.real(
        np.conj(density_hat) * density_hat
        + np.conj(dphi_dx_hat) * dphi_dx_hat
        + np.conj(dphi_dy_hat) * dphi_dy_hat
    ) / float(nx * ny) ** 2
    wavenumber = np.sqrt(kx_grid**2 + ky_grid**2)
    dk = min(2 * np.pi / lx, 2 * np.pi / ly)
    shell = np.rint(wavenumber / dk).astype(int)
    shell_energy = np.bincount(shell.ravel(), weights=energy.ravel())
    k = np.arange(shell_energy.size) * dk
    return k, shell_energy


def save_image(field, title, path, cmap=None, symmetric=False):
    nx, ny = field.shape
    fig, ax = plt.subplots(figsize=(6.3, 5.2))
    limits = {}
    if symmetric:
        maximum = float(np.max(np.abs(field)))
        if maximum > 0:
            limits = {"vmin": -maximum, "vmax": maximum}
    im = ax.imshow(
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


def save_spectrum_plot(
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
):
    spectra = {
        "DNS": isotropic_hw_energy_spectrum(dns, lx, ly),
        f"Filtered {filtered_fine.shape[-1]}": isotropic_hw_energy_spectrum(
            filtered_fine, lx, ly
        ),
        f"Filtered {coarse_n}": isotropic_hw_energy_spectrum(
            filtered_coarse, lx, ly
        ),
        inverse_input_label: isotropic_hw_energy_spectrum(
            predicted_filtered, lx, ly
        ),
        "Reconstructed DNS": isotropic_hw_energy_spectrum(
            predicted_dns, lx, ly
        ),
    }
    fig, ax = plt.subplots(figsize=(7.4, 5.6))
    for label, (k, p) in spectra.items():
        v = (k > 0) & (p > 0)
        if label == f"Filtered {coarse_n}":
            ax.loglog(
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
    ax.axvline(
        kn, linestyle=":", linewidth=1.4, label=f"{coarse_n}-grid Nyquist"
    )
    ax.set_title(
        r"HW energy spectrum: $E=\frac{1}{2}\int(n^2+|\nabla\phi|^2)$"
    )
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"Shell energy $E(k)$")
    ax.set_ylim(bottom=10e-10)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(numpoints=1)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
