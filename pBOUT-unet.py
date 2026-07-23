#!/usr/bin/env python3
"""BOUT++ interface for the Germano-filter Hasegawa--Wakatani U-Net.

The public functions intentionally match :mod:`pBOUT`: ``initFlow(npv)`` and
``findLESTerms(pLES)``.  BOUT++ supplies fields in (x, batch/y, z) order.  The
network consumes (batch, channel, x, z), reconstructs the fine/DNS fields, and
the returned nonlinear terms are again flattened in BOUT++ order.

Configuration is through environment variables so that the C++ embedding does
not need an argument parser::

  PBOUT_UNET_CHECKPOINT=/path/to/best.pt
  PBOUT_UNET_DNS_PATH=/path/to/dns_snapshot.npz
  PBOUT_UNET_DNS_INIT=1                  # run BOUT++ from provided DNS fields
  PBOUT_UNET_DEVICE=cuda                 # or cpu; default is automatic
  PBOUT_UNET_IMPLICIT=1                  # return -(DNS-LES) brackets
  PBOUT_UNET_VORTICITY_FROM_POTENTIAL=1  # enforce vort = Laplacian(phi)
  PBOUT_UNET_INVERSE_CUTOFF_FRACTION=0.75
  PBOUT_UNET_INVERSE_CUTOFF_ORDER=8
  PBOUT_UNET_POST_CUTOFF_FRACTION=0.5    # suppress post-inversion noise
  PBOUT_UNET_POST_CUTOFF_ORDER=8

The cutoff variables override checkpoint values.  Set the fraction to ``none``
to disable it.
"""

from __future__ import annotations

import os
import site
import sys
import shutil

from pathlib import Path

# BOUT++ embeds Python, so activating conda in the shell does not always add
# that environment's site-packages to the embedded interpreter.  Make the
# PyTorch environment explicit before importing torch.  This can be overridden
# for another installation without editing this file.
_DEFAULT_SITE_PACKAGES = (
    "/gpfs/mcc/HT04543/jxc06/jxc74-jxc06/programs/miniconda3/envs/py311/"
    "lib/python3.11/site-packages"
)
_SITE_PACKAGES = os.environ.get(
    "PBOUT_UNET_SITE_PACKAGES", _DEFAULT_SITE_PACKAGES
)
if _SITE_PACKAGES and sys.version_info[:2] != (3, 11):
    raise RuntimeError(
        "pBOUT-unet uses the py311 PyTorch environment, but the BOUT++ "
        f"executable embeds Python {sys.version_info.major}."
        f"{sys.version_info.minor}. Rebuild hasegawa-wakatani against "
        "/gpfs/mcc/HT04543/jxc06/jxc74-jxc06/programs/miniconda3/envs/"
        "py311/lib/libpython3.11.so, or set PBOUT_UNET_SITE_PACKAGES to a "
        "PyTorch installation matching the embedded Python version."
    )
if _SITE_PACKAGES and _SITE_PACKAGES not in sys.path:
    site.addsitedir(_SITE_PACKAGES)

import numpy as np
import torch
import matplotlib.pyplot as plt

from germano_common import (
    ResidualUNetSR,
    filter_snapshot,
    hw_fields_torch,
    inverse_germano_torch,
    isotropic_spectrum,
    load_dns_snapshot,
    load_checkpoint,
    periodic_laplacian_4th_torch,
    save_image,
    spectral_cutoff_torch,
    validate_hw_checkpoint,
)
from visualization_common import (
    arakawa_bracket as shared_arakawa_bracket,
    save_arakawa_comparison,
)


_HERE = Path(__file__).resolve().parent
_DEFAULT_CHECKPOINT = (
    _HERE / "germano_training_output_32_v4" / "checkpoints" / "best.pt"
)
CHECKPOINT = Path(os.environ.get("PBOUT_UNET_CHECKPOINT", _DEFAULT_CHECKPOINT))
_dns_path = os.environ.get("PBOUT_UNET_DNS_PATH")
DNS_PATH = Path(_dns_path).expanduser() if _dns_path else None
DNS_INIT = os.environ.get("PBOUT_UNET_DNS_INIT", "0").lower() not in {
    "0", "false", "no", "off"
}
IMPLICIT = os.environ.get("PBOUT_UNET_IMPLICIT", "1").lower() not in {
    "0", "false", "no", "off"
}
VORTICITY_FROM_POTENTIAL = os.environ.get(
    "PBOUT_UNET_VORTICITY_FROM_POTENTIAL", "1"
).lower() not in {"0", "false", "no", "off"}


def _optional_float(name: str, default):
    value = os.environ.get(name)
    if value is None:
        return default
    if value.lower() in {"none", "off", "disable", "disabled"}:
        return None
    return float(value)


_device_name = os.environ.get(
    "PBOUT_UNET_DEVICE", "cuda" if torch.cuda.is_available() else "cpu"
)
DEVICE = torch.device(_device_name)
if DEVICE.type == "cuda" and not torch.cuda.is_available():
    raise RuntimeError("PBOUT_UNET_DEVICE requests CUDA, but CUDA is unavailable")
if not CHECKPOINT.is_file():
    raise FileNotFoundError(
        f"U-Net checkpoint not found: {CHECKPOINT}. Set PBOUT_UNET_CHECKPOINT."
    )

_checkpoint = load_checkpoint(CHECKPOINT, DEVICE)
validate_hw_checkpoint(_checkpoint)
FIELD_NAMES = tuple(_checkpoint["field_names"])
FINE_N = int(_checkpoint["fine_n"])
COARSE_N = int(_checkpoint["coarse_n"])
LX = float(_checkpoint["lx"])
LY = float(_checkpoint["ly"])
ALPHA = float(_checkpoint["alpha"])
BASE_CHANNELS = int(_checkpoint["base_channels"])
PRECISION = _checkpoint.get("precision", "float32")
if PRECISION not in ("float32", "float64"):
    raise ValueError(f"Unsupported checkpoint precision: {PRECISION}")
TORCH_DTYPE = getattr(torch, PRECISION)
NUMPY_DTYPE = getattr(np, PRECISION)
VORTICITY_MODE = _checkpoint.get("vorticity_mode", "potential_laplacian")
LEARN_VORTICITY = VORTICITY_MODE == "learned_channel"
OUTPUT_CHANNELS = int(
    _checkpoint.get("model_output_channels", 3 if LEARN_VORTICITY else 2)
)
if FINE_N % COARSE_N:
    raise ValueError("Checkpoint fine_n must be divisible by coarse_n")

MODEL = ResidualUNetSR(
    len(FIELD_NAMES), BASE_CHANNELS, FINE_N // COARSE_N, OUTPUT_CHANNELS
).to(device=DEVICE, dtype=TORCH_DTYPE)
MODEL.load_state_dict(_checkpoint["model"])
MODEL.eval()
MODEL.requires_grad_(False)

MEAN = torch.as_tensor(_checkpoint["mean"], dtype=TORCH_DTYPE, device=DEVICE)
STD = torch.as_tensor(_checkpoint["std"], dtype=TORCH_DTYPE, device=DEVICE)
CUTOFF_FRACTION = _optional_float(
    "PBOUT_UNET_INVERSE_CUTOFF_FRACTION",
    _checkpoint.get("inverse_cutoff_fraction", 0.75),
)
CUTOFF_ORDER = float(
    os.environ.get(
        "PBOUT_UNET_INVERSE_CUTOFF_ORDER",
        _checkpoint.get("inverse_cutoff_order", 8.0),
    )
)
POST_CUTOFF_FRACTION = _optional_float(
    "PBOUT_UNET_POST_CUTOFF_FRACTION", 0.5
)
POST_CUTOFF_ORDER = float(
    os.environ.get("PBOUT_UNET_POST_CUTOFF_ORDER", 8.0)
)

# Previous-step amplitudes used to keep reconstructed DNS fields on the
# evolving LES amplitude scale.  initFlow supplies the first pair.
_PREVIOUS_LES_AMAX = None
_PREVIOUS_DNS_AMAX = None


def _arakawa_bracket(f: torch.Tensor, g: torch.Tensor, dx: float, dy: float):
    """Compatibility wrapper for the shared Arakawa implementation."""
    return shared_arakawa_bracket(f, g, dx, dy)


def _germano_filter_torch(field: torch.Tensor) -> torch.Tensor:
    """Apply the checkpoint's forward differential/Germano filter."""
    nx, ny = field.shape[-2:]
    kx = 2 * torch.pi * torch.fft.fftfreq(
        nx, d=LX / nx, device=field.device, dtype=field.dtype
    )
    ky = 2 * torch.pi * torch.fft.fftfreq(
        ny, d=LY / ny, device=field.device, dtype=field.dtype
    )
    kxg, kyg = torch.meshgrid(kx, ky, indexing="ij")
    multiplier = 1.0 / (1.0 + ALPHA**2 * (kxg.square() + kyg.square()))
    return torch.fft.ifft2(
        torch.fft.fft2(field, dim=(-2, -1)) * multiplier,
        dim=(-2, -1),
    ).real


def _spectral_resize_torch(field: torch.Tensor, target_n: int) -> torch.Tensor:
    """Periodic Fourier resize, equivalent to germano_common.spectral_resize."""
    source_n = field.shape[-1]
    if field.shape[-2] != source_n:
        raise ValueError("Only square x-z fields are supported")
    shifted = torch.fft.fftshift(
        torch.fft.fft2(field, dim=(-2, -1)), dim=(-2, -1)
    )
    copy_n = min(source_n, target_n)
    source_start = (source_n - copy_n) // 2
    target_start = (target_n - copy_n) // 2
    target = torch.zeros(
        (*field.shape[:-2], target_n, target_n),
        dtype=shifted.dtype,
        device=field.device,
    )
    target[..., target_start:target_start + copy_n,
           target_start:target_start + copy_n] = shifted[
        ..., source_start:source_start + copy_n,
        source_start:source_start + copy_n
    ]
    target *= (target_n / source_n) ** 2
    return torch.fft.ifft2(
        torch.fft.ifftshift(target, dim=(-2, -1)), dim=(-2, -1)
    ).real


def _decode_fields(values: np.ndarray, offset: int, required_channels: int):
    payload = values[offset:]
    plane = COARSE_N * COARSE_N
    # The current hw.cxx always allocates five blocks, even in explicit mode;
    # also accept a future compact three-block explicit payload.
    stored_channels = 5 if payload.size % (5 * plane) == 0 else 3
    if stored_channels < required_channels or payload.size % (stored_channels * plane):
        raise ValueError(
            f"BOUT payload has {payload.size} values; expected a multiple of "
            f"{required_channels * plane} for a {COARSE_N}x{COARSE_N} grid"
        )
    batch = payload.size // (stored_channels * plane)
    arrays = []
    for channel in range(required_channels):
        block = payload[channel * batch * plane:(channel + 1) * batch * plane]
        # C++ block order is (x, batch/y, z).
        arrays.append(block.reshape(COARSE_N, batch, COARSE_N).transpose(1, 0, 2))
    return np.stack(arrays, axis=1), batch


@torch.inference_mode()
def _reconstruct_dns(coarse: torch.Tensor):
    """Return reconstructed DNS fields and the cutoff-filtered U-Net output."""
    normalised = (coarse - MEAN) / STD
    predicted = MODEL(normalised)
    if LEARN_VORTICITY:
        filtered_fine = predicted * STD + MEAN
    else:
        uv = predicted * STD[:, :2] + MEAN[:, :2]
        filtered_fine = hw_fields_torch(uv[:, :1], uv[:, 1:2], LX, LY)
    inverse_input = spectral_cutoff_torch(
        filtered_fine, LX, LY, CUTOFF_FRACTION, CUTOFF_ORDER
    )
    dns = inverse_germano_torch(inverse_input, ALPHA, LX, LY)
    dns = spectral_cutoff_torch(
        dns, LX, LY, POST_CUTOFF_FRACTION, POST_CUTOFF_ORDER
    )
    if VORTICITY_FROM_POTENTIAL:
        vorticity = periodic_laplacian_4th_torch(
            dns[:, 1:2], LX, LY
        )
        dns = torch.cat((dns[:, :2], vorticity), dim=1)
    return dns, inverse_input


def _validate_les_spacing(dx: float, dy: float) -> None:
    """Ensure BOUT++ and the trained checkpoint describe the same domain."""
    bout_lx = dx * COARSE_N
    bout_ly = dy * COARSE_N
    if not np.isclose(bout_lx, LX, rtol=1e-10, atol=1e-12) or not np.isclose(
        bout_ly, LY, rtol=1e-10, atol=1e-12
    ):
        raise ValueError(
            "BOUT++ domain does not match the U-Net training domain: "
            f"BOUT++ Lx={bout_lx}, Ly={bout_ly}; checkpoint Lx={LX}, Ly={LY}."
        )


def _save_arakawa_images(
    time_step: int,
    les_vorticity: torch.Tensor,
    les_density: torch.Tensor,
    dns_vorticity: torch.Tensor,
    dns_density: torch.Tensor,
    filtered_vorticity: torch.Tensor,
    filtered_density: torch.Tensor,
) -> None:
    """Save LES, DNS, and filtered-DNS terms in diagnostic layouts."""
    output_dir = Path(f"results_StylES_{COARSE_N}x{COARSE_N}") / "arakawa"
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostic_sets = (
        (
            "LES",
            "LES input Arakawa terms",
            les_vorticity.detach().cpu().double().numpy(),
            les_density.detach().cpu().double().numpy(),
        ),
        (
            "DNS",
            "DNS-resolution Arakawa terms before filtering",
            dns_vorticity.detach().cpu().double().numpy(),
            dns_density.detach().cpu().double().numpy(),
        ),
        (
            "filtered_DNS",
            "Filtered DNS Arakawa terms",
            filtered_vorticity.detach().cpu().double().numpy(),
            filtered_density.detach().cpu().double().numpy(),
        ),
    )
    term_titles = (r"$J(\phi,\omega)$", r"$J(\phi,n)$")

    for name, figure_title, vorticity_arrays, density_arrays in diagnostic_sets:
        batch_count = len(vorticity_arrays)
        for batch_index, fields in enumerate(
            zip(vorticity_arrays, density_arrays)
        ):
            nx, nz = fields[0].shape
            center_index = nz // 2
            x_pixels = np.arange(nx)
            fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.2))
            for column, (field, title) in enumerate(zip(fields, term_titles)):
                maximum = float(np.max(np.abs(field)))
                limits = (
                    {"vmin": -maximum, "vmax": maximum}
                    if maximum > 0 else {}
                )
                image = axes[0, column].imshow(
                    field.T,
                    origin="lower",
                    extent=(0, nx, 0, nz),
                    aspect="equal",
                    cmap="coolwarm",
                    **limits,
                )
                axes[0, column].set_title(f"{figure_title} {title}")
                axes[0, column].set_xlabel("x [pixels]")
                axes[0, column].set_ylabel("z [pixels]")
                fig.colorbar(image, ax=axes[0, column])

                axes[1, column].plot(
                    x_pixels, field[:, center_index], linewidth=1.2
                )
                axes[1, column].axhline(
                    0.0, color="0.5", linewidth=0.7
                )
                axes[1, column].set_title(
                    f"{title}: centerline z={center_index}"
                )
                axes[1, column].set_xlabel("x [pixels]")
                axes[1, column].set_ylabel("Arakawa term")
                axes[1, column].grid(True, alpha=0.25)

            fig.suptitle(f"{figure_title}, step {time_step}")
            fig.tight_layout()
            batch_suffix = (
                "" if batch_count == 1 else f"_batch{batch_index:03d}"
            )
            fig.savefig(
                output_dir
                / f"arakawa_{name}_terms_time{time_step:07d}"
                  f"{batch_suffix}.png",
                dpi=180,
            )
            plt.close(fig)


def _save_les_bout_arakawa_comparison(
    time_step: int,
    calculated_vorticity: torch.Tensor,
    calculated_density: torch.Tensor,
    bout_vorticity: torch.Tensor,
    bout_density: torch.Tensor,
) -> None:
    """Compare Python and BOUT++ Arakawa terms, including centerlines."""
    output_dir = (
        Path(f"results_StylES_{COARSE_N}x{COARSE_N}") / "arakawa"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = (
        (
            "vorticity",
            calculated_vorticity,
            bout_vorticity,
            r"$J(V,P)$",
        ),
        (
            "density",
            calculated_density,
            bout_density,
            r"$J(V,U)$",
        ),
    )
    center_index = COARSE_N // 2
    x_pixels = np.arange(COARSE_N)

    for name, calculated, bout, term_label in diagnostics:
        calculated_arrays = calculated.detach().cpu().double().numpy()
        bout_arrays = bout.detach().cpu().double().numpy()
        for batch_index, (calculated_field, bout_field) in enumerate(
            zip(calculated_arrays, bout_arrays)
        ):
            difference = calculated_field - bout_field
            fields_to_plot = (
                (calculated_field, "Calculated in pBOUT-unet"),
                (bout_field, "Provided by BOUT++"),
                (difference, "Calculated - BOUT++"),
            )
            fig, axes = plt.subplots(2, 3, figsize=(17.5, 9.2))
            for column, (values, label) in enumerate(fields_to_plot):
                maximum = float(np.max(np.abs(values)))
                limits = (
                    {"vmin": -maximum, "vmax": maximum}
                    if maximum > 0 else {}
                )
                image = axes[0, column].imshow(
                    values.T,
                    origin="lower",
                    extent=(0, COARSE_N, 0, COARSE_N),
                    aspect="equal",
                    cmap="coolwarm",
                    **limits,
                )
                axes[0, column].set_title(f"{term_label}: {label}")
                axes[0, column].set_xlabel("x [pixels]")
                axes[0, column].set_ylabel("y [pixels]")
                fig.colorbar(image, ax=axes[0, column])

                axes[1, column].plot(
                    x_pixels, values[:, center_index], linewidth=1.2
                )
                axes[1, column].axhline(
                    0.0, color="0.5", linewidth=0.7
                )
                axes[1, column].set_title(
                    f"{label}: centerline y={center_index}"
                )
                axes[1, column].set_xlabel("x [pixels]")
                axes[1, column].set_ylabel("Arakawa term")
                axes[1, column].grid(True, alpha=0.25)

            fig.suptitle(
                f"LES Arakawa implementation comparison, step {time_step}",
                fontsize=14,
            )
            fig.tight_layout()
            batch_suffix = (
                "" if len(calculated_arrays) == 1
                else f"_batch{batch_index:03d}"
            )
            fig.savefig(
                output_dir
                / f"arakawa_LES_{name}_comparison_time{time_step:07d}"
                  f"{batch_suffix}.png",
                dpi=180,
            )
            plt.close(fig)


def saveArakawaTerms(pArakawa):
    """Write BOUT++ Arakawa terms and a centerline image for one time step.

    The flat payload is ``step, time, nx, nz, J(phi,vort), J(phi,n)``.  The
    two fields use the same ``(x, y/batch, z)`` ordering as the other BOUT++
    callbacks; this diagnostic currently requires the HW model's single
    y-plane layout.
    """

    values = np.asarray(pArakawa, dtype=np.float64).reshape(-1)
    if values.size < 4:
        raise ValueError("saveArakawaTerms requires a four-value header")

    time_step = int(values[0])
    simtime = float(values[1])
    nx = int(values[2])
    nz = int(values[3])
    plane = nx * nz
    if nx <= 0 or nz <= 0 or values.size != 4 + 2 * plane:
        raise ValueError(
            f"saveArakawaTerms received {values.size} values; expected "
            f"{4 + 2 * plane} for a {nx}x{nz} grid"
        )

    terms = values[4:].reshape(2, nx, nz)
    output_dir = Path("results_unet_DNS")
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"arakawa_terms_time{time_step:07d}"
    np.savez_compressed(
        output_dir / f"{stem}.npz",
        pStep=np.asarray(time_step),
        simtime=np.asarray(simtime),
        phi_vorticity=terms[0],
        phi_density=terms[1],
    )

    center_index = nz // 2
    x_pixels = np.arange(nx)
    diagnostics = (
        (terms[0], r"$J(\phi,\omega)$"),
        (terms[1], r"$J(\phi,n)$"),
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.2))
    for column, (field, title) in enumerate(diagnostics):
        maximum = float(np.max(np.abs(field)))
        limits = {"vmin": -maximum, "vmax": maximum} if maximum > 0 else {}
        image = axes[0, column].imshow(
            field.T,
            origin="lower",
            extent=(0, nx, 0, nz),
            aspect="equal",
            cmap="coolwarm",
            **limits,
        )
        axes[0, column].set_title(f"BOUT++ Arakawa term {title}")
        axes[0, column].set_xlabel("x [pixels]")
        axes[0, column].set_ylabel("z [pixels]")
        fig.colorbar(image, ax=axes[0, column])

        axes[1, column].plot(x_pixels, field[:, center_index], linewidth=1.2)
        axes[1, column].axhline(0.0, color="0.5", linewidth=0.7)
        axes[1, column].set_title(f"{title}: centerline z={center_index}")
        axes[1, column].set_xlabel("x [pixels]")
        axes[1, column].set_ylabel("Arakawa term")
        axes[1, column].grid(True, alpha=0.25)

    fig.suptitle(f"DNS Arakawa terms, step {time_step}, t={simtime:g}")
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=180)
    plt.close(fig)
    return None


@torch.inference_mode()
def _save_initial_dns_arakawa(
    fields: np.ndarray,
    reference_fields: np.ndarray | None = None,
) -> tuple[Path, Path]:
    """Calculate and save HW Arakawa fields and spectra on the DNS grid."""
    if reference_fields is not None:
        return save_arakawa_comparison(
            fields,
            reference_fields,
            LX,
            LY,
            Path(f"results_StylES_{COARSE_N}x{COARSE_N}") / "arakawa",
            stem="arakawa_DNS_time0000",
            cutoff_fraction=POST_CUTOFF_FRACTION,
        )
    dx_dns = LX / FINE_N
    dy_dns = LY / FINE_N

    def calculate_terms(source: np.ndarray):
        dns = torch.from_numpy(source[None]).to(
            device=DEVICE, dtype=TORCH_DTYPE
        )
        potential = dns[:, 1:2]
        return (
            _arakawa_bracket(potential, dns[:, 2:3], dx_dns, dy_dns)[0, 0],
            _arakawa_bracket(potential, dns[:, 0:1], dx_dns, dy_dns)[0, 0],
        )

    current_terms = calculate_terms(fields)
    reference_terms = (
        calculate_terms(reference_fields) if reference_fields is not None else None
    )
    titles = (r"DNS $J(V,P)$", r"DNS $J(V,U)$")

    output_dir = (
        Path(f"results_StylES_{COARSE_N}x{COARSE_N}") / "arakawa"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "arakawa_DNS_time0000.png"

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.2))
    center_index = FINE_N // 2
    x_pixels = np.arange(FINE_N)
    for term_index, (ax, term, title) in enumerate(
        zip(axes[0], current_terms, titles)
    ):
        values = term.detach().cpu().double().numpy()
        maximum = float(np.max(np.abs(values)))
        limits = {"vmin": -maximum, "vmax": maximum} if maximum > 0 else {}
        image = ax.imshow(
            values.T,
            origin="lower",
            extent=(0, FINE_N, 0, FINE_N),
            aspect="equal",
            cmap="coolwarm",
            **limits,
        )
        ax.set_title(title)
        ax.set_xlabel("x [pixels]")
        ax.set_ylabel("y [pixels]")
        fig.colorbar(image, ax=ax)
        center_ax = axes[1, term_index]
        current_label = "Reconstructed DNS" if reference_terms else "DNS"
        center_ax.plot(
            x_pixels,
            values[:, center_index],
            color="tab:blue",
            linewidth=1.2,
            label=current_label,
        )
        if reference_terms is not None:
            reference_values = (
                reference_terms[term_index].detach().cpu().double().numpy()
            )
            center_ax.plot(
                x_pixels,
                reference_values[:, center_index],
                color="black",
                linewidth=1.0,
                label="Original DNS",
            )
        center_ax.axhline(0.0, color="0.5", linewidth=0.7)
        center_ax.set_title(f"{title}: centerline y={center_index}")
        center_ax.set_xlabel("x [pixels]")
        center_ax.set_ylabel("Arakawa term")
        center_ax.grid(True, alpha=0.25)
        center_ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    spectrum_path = output_dir / "arakawa_DNS_spectrum_time0000.png"
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2))
    for term_index, (ax, term, title) in enumerate(
        zip(axes, current_terms, titles)
    ):
        values = term.detach().cpu().double().numpy()
        wavenumber, power = isotropic_spectrum(values, LX, LY)
        positive = (wavenumber > 0) & (power > 0)
        current_label = "Reconstructed DNS" if reference_terms else "DNS"
        ax.loglog(
            wavenumber[positive], power[positive],
            color="tab:blue", label=current_label,
        )
        if reference_terms is not None:
            reference_values = reference_terms[term_index].detach().cpu().double().numpy()
            reference_k, reference_power = isotropic_spectrum(
                reference_values, LX, LY
            )
            reference_positive = (reference_k > 0) & (reference_power > 0)
            ax.loglog(
                reference_k[reference_positive],
                reference_power[reference_positive],
                color="black",
                label="Original DNS",
            )
        if POST_CUTOFF_FRACTION is not None:
            grid_nyquist = min(
                np.pi / (LX / FINE_N), np.pi / (LY / FINE_N)
            )
            cutoff = POST_CUTOFF_FRACTION * grid_nyquist
            ax.axvline(
                cutoff,
                color="tab:red",
                linestyle="--",
                label=f"post cutoff ({POST_CUTOFF_FRACTION:g} Nyquist)",
            )
        ax.legend()
        ax.set_title(f"{title} spectrum")
        ax.set_xlabel(r"wavenumber $k$")
        ax.set_ylabel("shell-summed power")
        ax.set_ylim(bottom=10e-10)
        ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(spectrum_path, dpi=180)
    plt.close(fig)
    return output_path, spectrum_path


def _save_initial_dns_fields(
    fields: np.ndarray,
    reference_fields: np.ndarray | None = None,
) -> tuple[Path, Path]:
    """Save DNS fields and their original/reconstructed spectra."""
    output_dir = Path(f"results_StylES_{COARSE_N}x{COARSE_N}") / "fields"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "fields_DNS_time0000.png"
    diagnostics = (
        (fields[0], r"DNS density $U$"),
        (fields[1], r"DNS electrostatic potential $V$"),
        (fields[2], r"DNS vorticity $P$"),
    )

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2))
    for ax, (values, title) in zip(axes, diagnostics):
        image = ax.imshow(
            values.T,
            origin="lower",
            extent=(0, FINE_N, 0, FINE_N),
            aspect="equal",
            cmap="viridis",
        )
        ax.set_title(title)
        ax.set_xlabel("x [pixels]")
        ax.set_ylabel("y [pixels]")
        fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

    spectrum_path = output_dir / "fields_DNS_spectrum_time0000.png"
    spectrum_titles = (
        r"Density $U$ spectrum",
        r"Electrostatic potential $V$ spectrum",
        r"Vorticity $P$ spectrum",
    )
    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.2))
    for channel, (ax, title) in enumerate(zip(axes, spectrum_titles)):
        wavenumber, power = isotropic_spectrum(fields[channel], LX, LY)
        positive = (wavenumber > 0) & (power > 0)
        current_label = "Reconstructed DNS" if reference_fields is not None else "DNS"
        ax.loglog(
            wavenumber[positive],
            power[positive],
            color="tab:blue",
            label=current_label,
        )
        if reference_fields is not None:
            reference_k, reference_power = isotropic_spectrum(
                reference_fields[channel], LX, LY
            )
            reference_positive = (reference_k > 0) & (reference_power > 0)
            ax.loglog(
                reference_k[reference_positive],
                reference_power[reference_positive],
                color="black",
                label="Original DNS",
            )
        if POST_CUTOFF_FRACTION is not None:
            grid_nyquist = min(
                np.pi / (LX / FINE_N), np.pi / (LY / FINE_N)
            )
            ax.axvline(
                POST_CUTOFF_FRACTION * grid_nyquist,
                color="tab:red",
                linestyle="--",
                label=f"post cutoff ({POST_CUTOFF_FRACTION:g} Nyquist)",
            )
        ax.set_title(title)
        ax.set_xlabel(r"wavenumber $k$")
        ax.set_ylabel("shell-summed power")
        ax.set_ylim(bottom=10e-10)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(spectrum_path, dpi=180)
    plt.close(fig)
    return output_path, spectrum_path


def _load_dns_uvp(path: Path) -> np.ndarray:
    """Load the original U, V, and P arrays at checkpoint DNS resolution."""
    with np.load(path) as snapshot:
        missing = [name for name in ("U", "V", "P") if name not in snapshot]
        if missing:
            raise KeyError(f"{path} is missing fields: {missing}")
        fields = np.stack(
            [
                np.asarray(snapshot[name], dtype=NUMPY_DTYPE)
                for name in ("U", "V", "P")
            ]
        )
    expected_dns_shape = (3, FINE_N, FINE_N)
    if fields.shape != expected_dns_shape:
        raise ValueError(
            f"{path}: fields U, V, P have combined shape {fields.shape}, "
            f"expected {expected_dns_shape}"
        )
    if not np.all(np.isfinite(fields)):
        raise ValueError(f"{path}: fields U, V, P contain NaN or Inf")
    return fields


def initFlow(npv):
    """Return initial LES fields, optionally derived from a DNS snapshot.

    If ``PBOUT_UNET_DNS_INIT`` is enabled, ``PBOUT_UNET_DNS_PATH`` must point
    to a snapshot at the checkpoint's DNS resolution.  Its U, V, and P fields
    are returned unchanged so BOUT++ can run at DNS resolution, without
    calling the U-Net.

    If ``PBOUT_UNET_DNS_PATH`` is set, load the ``U`` (density) and ``V``
    (potential) arrays from that ``.npz`` DNS snapshot, apply the checkpoint's
    Germano filter, and spectrally resize them to the LES grid.  Otherwise,
    preserve the initial fields supplied by BOUT++.
    """

    global _PREVIOUS_LES_AMAX, _PREVIOUS_DNS_AMAX

    if DNS_INIT:
        if DNS_PATH is None:
            raise ValueError(
                "PBOUT_UNET_DNS_INIT requires PBOUT_UNET_DNS_PATH to be set."
            )

        if not DNS_PATH.is_file():
            raise FileNotFoundError(
                f"DNS snapshot not found: {DNS_PATH}. "
                "Set PBOUT_UNET_DNS_PATH to an existing .npz file."
            )
        fields = _load_dns_uvp(DNS_PATH)

        fields_output, fields_spectrum_output = _save_initial_dns_fields(fields)
        arakawa_output, arakawa_spectrum_output = (
            _save_initial_dns_arakawa(fields)
        )

        print(
            f"Initialized BOUT++ directly from DNS snapshot {DNS_PATH};\n"
            f"resolution={FINE_N}, Lx={LX}, Ly={LY};\n"
            "fields were not filtered or resized and U-Net was not called;\n"
            f"saved DNS fields to {fields_output};\n"
            f"saved their spectra to {fields_spectrum_output};\n"
            f"saved DNS Arakawa terms to {arakawa_output};\n"
            f"saved their spectra to {arakawa_spectrum_output}\n",
            flush=True,
        )
        return fields.reshape(-1).astype(np.float64, copy=False)

    reference_dns = None
    if DNS_PATH is None:
        values = np.asarray(npv, dtype=np.float64).reshape(-1)
        expected = 2 + 3 * COARSE_N * COARSE_N
        if values.size != expected:
            raise ValueError(
                f"initFlow received {values.size} values, expected {expected}. "
                "In hw.cxx set initFlow's NumPy dimension to 2 + 3*SIZE*SIZE."
            )
        _validate_les_spacing(float(values[0]), float(values[1]))

        # initFlow's C++ payload is point-interleaved n, phi, vort.
        fields = values[2:].reshape(COARSE_N, COARSE_N, 3).transpose(2, 0, 1)
        initial_state = "BOUT++ fields"
    else:
        if not DNS_PATH.is_file():
            raise FileNotFoundError(
                f"DNS snapshot not found: {DNS_PATH}. "
                "Set PBOUT_UNET_DNS_PATH to an existing .npz file."
            )
        reference_dns = _load_dns_uvp(DNS_PATH)
        dns = load_dns_snapshot(
            DNS_PATH, FIELD_NAMES, FINE_N, LX, LY, dtype=NUMPY_DTYPE
        )
        fields, _ = filter_snapshot(
            dns, ALPHA, LX, LY, COARSE_N, dtype=NUMPY_DTYPE
        )
        initial_state = f"filtered DNS snapshot {DNS_PATH}"

    # Save the DNS reconstruction of the exact LES state returned to BOUT++.
    coarse = torch.from_numpy(fields[None]).to(
        device=DEVICE, dtype=TORCH_DTYPE
    )
    initial_dns, initial_fdns = _reconstruct_dns(coarse)
    _PREVIOUS_LES_AMAX = torch.amax(
        torch.abs(coarse), dim=(-2, -1), keepdim=True
    )
    _PREVIOUS_DNS_AMAX = torch.amax(
        torch.abs(initial_dns), dim=(-2, -1), keepdim=True
    )
    if torch.any(_PREVIOUS_LES_AMAX == 0):
        zero_channels = torch.nonzero(
            _PREVIOUS_LES_AMAX[0, :, 0, 0] == 0, as_tuple=False
        ).flatten().cpu().tolist()
        raise ValueError(
            "Cannot initialize DNS amplitude rescaling because the initial "
            f"LES absolute maximum is zero for channel(s) {zero_channels}."
        )
    initial_dns = initial_dns[0].detach().cpu().double().numpy()
    initial_fdns = initial_fdns[0].detach().cpu().double().numpy()
    results_dir = Path(f"results_StylES_{COARSE_N}x{COARSE_N}") / "fields"
    if results_dir.exists():
        shutil.rmtree(results_dir)
    results_dir.mkdir(parents=True)
    time_step = 0
    initial_output = results_dir / "fields_DNS_0000000.npz"
    np.savez_compressed(
        initial_output,
        pStep=np.asarray(0),
        simtime=np.asarray(0.0),
        U=initial_dns[0],
        V=initial_dns[1],
        P=initial_dns[2],
    )
    initial_filtered_output = results_dir / "fields_fDNS_0000000.npz"
    np.savez_compressed(
        initial_filtered_output,
        pStep=np.asarray(0),
        simtime=np.asarray(0.0),
        U=initial_fdns[0],
        V=initial_fdns[1],
        P=initial_fdns[2],
    )
    fields_image_output, fields_spectrum_output = _save_initial_dns_fields(
        initial_dns, reference_dns
    )
    arakawa_image_output, arakawa_spectrum_output = (
        _save_initial_dns_arakawa(initial_dns, reference_dns)
    )

    print(
        f"Loaded BOUT++ U-Net checkpoint {CHECKPOINT} on {DEVICE};\n"
        f"LES={COARSE_N}, DNS={FINE_N}, implicit={IMPLICIT};\n"
        f"precision={PRECISION}, Lx={LX}, Ly={LY};\n"
        f"vorticity_from_potential={VORTICITY_FROM_POTENTIAL};\n"
        f"initial state={initial_state}; saved reconstruction={initial_output};\n"
        f"saved U-Net output={initial_filtered_output};\n"
        f"saved DNS fields image={fields_image_output};\n"
        f"saved DNS fields spectrum={fields_spectrum_output};\n"
        f"saved DNS Arakawa image={arakawa_image_output};\n"
        f"saved DNS Arakawa spectrum={arakawa_spectrum_output}\n",
        flush=True,
    )

    return fields.reshape(-1).astype(np.float64, copy=False)


@torch.inference_mode()
def findLESTerms(pLES):
    global _PREVIOUS_LES_AMAX, _PREVIOUS_DNS_AMAX

    values = np.asarray(pLES, dtype=np.float64).reshape(-1)
    if values.size < 4:
        raise ValueError("findLESTerms requires the four-value BOUT header")
    time_step = int(values[0])
    simtime = float(values[3])
    dx_les = float(values[2])
    _validate_les_spacing(dx_les, dx_les)
    required_channels = 5 if IMPLICIT else 3
    fields, batch = _decode_fields(values, 4, required_channels)

    coarse = torch.from_numpy(fields[:, :3]).to(
        device=DEVICE, dtype=TORCH_DTYPE
    )
    dns, filtered_fine = _reconstruct_dns(coarse)
    if _PREVIOUS_LES_AMAX is None or _PREVIOUS_DNS_AMAX is None:
        raise RuntimeError(
            "findLESTerms was called before initFlow initialized the LES/DNS "
            "amplitude references"
        )
    les_amax = torch.amax(torch.abs(coarse), dim=(-2, -1), keepdim=True)
    target_dns_amax = (
        les_amax / _PREVIOUS_LES_AMAX * _PREVIOUS_DNS_AMAX
    )
    reconstructed_dns_amax = torch.amax(
        torch.abs(dns), dim=(-2, -1), keepdim=True
    )
    dns = torch.where(
        reconstructed_dns_amax > 0,
        dns * target_dns_amax / reconstructed_dns_amax.clamp_min(
            torch.finfo(dns.dtype).tiny
        ),
        dns,
    )
    _PREVIOUS_LES_AMAX = les_amax
    _PREVIOUS_DNS_AMAX = target_dns_amax
    if time_step % 100 == 0:
        results_dir = Path(
            f"results_StylES_{COARSE_N}x{COARSE_N}"
        ) / "fields"
        results_dir.mkdir(parents=True, exist_ok=True)
        dns_np = dns.detach().cpu().double().numpy()
        output_fields = (
            dns_np[0] if batch == 1 else dns_np.transpose(1, 0, 2, 3)
        )
        np.savez_compressed(
            results_dir / f"fields_DNS_{(time_step+1):07d}.npz",
            pStep=np.asarray(time_step),
            simtime=np.asarray(simtime),
            U=output_fields[0],
            V=output_fields[1],
            P=output_fields[2],
        )
        filtered_np = filtered_fine.detach().cpu().double().numpy()
        filtered_output_fields = (
            filtered_np[0]
            if batch == 1
            else filtered_np.transpose(1, 0, 2, 3)
        )
        # np.savez_compressed(
        #     results_dir / f"fields_fDNS_{(time_step+1):07d}.npz",
        #     pStep=np.asarray(time_step),
        #     simtime=np.asarray(simtime),
        #     U=filtered_output_fields[0],
        #     V=filtered_output_fields[1],
        #     P=filtered_output_fields[2],
        # )

    dx_dns = LX / FINE_N
    dy_dns = LY / FINE_N
    phi = dns[:, 1:2]
    bracket_vort = _arakawa_bracket(phi, dns[:, 2:3], dx_dns, dy_dns)
    bracket_n = _arakawa_bracket(phi, dns[:, 0:1], dx_dns, dy_dns)
    dns_bracket_vort = bracket_vort[:, 0]
    dns_bracket_n = bracket_n[:, 0]

    dx_les_checkpoint = LX / COARSE_N
    dy_les_checkpoint = LY / COARSE_N
    les_phi = coarse[:, 1:2]
    les_bracket_vort = _arakawa_bracket(
        les_phi, coarse[:, 2:3], dx_les_checkpoint, dy_les_checkpoint
    )[:, 0]
    les_bracket_n = _arakawa_bracket(
        les_phi, coarse[:, 0:1], dx_les_checkpoint, dy_les_checkpoint
    )[:, 0]

    # misc_styles.find_bracket filters the fine-grid bracket before returning it
    # to LES.  Use the same Germano operator and the training spectral coarsening.
    bracket_vort = _spectral_resize_torch(
        _germano_filter_torch(bracket_vort), COARSE_N
    )[:, 0]
    bracket_n = _spectral_resize_torch(
        _germano_filter_torch(bracket_n), COARSE_N
    )[:, 0]

    # if time_step % 1 == 0:
    #     _save_arakawa_images(
    #         time_step,
    #         les_bracket_vort,
    #         les_bracket_n,
    #         dns_bracket_vort,
    #         dns_bracket_n,
    #         bracket_vort,
    #         bracket_n,
    #     )

    if IMPLICIT:
        les_vort = torch.from_numpy(fields[:, 3]).to(
            device=DEVICE, dtype=TORCH_DTYPE
        )
        les_n = torch.from_numpy(fields[:, 4]).to(
            device=DEVICE, dtype=TORCH_DTYPE
        )
        # if time_step % 1000 == 0:
        #     _save_les_bout_arakawa_comparison(
        #         time_step,
        #         les_bracket_vort,
        #         les_bracket_n,
        #         les_vort,
        #         les_n,
        #     )
        bracket_vort = -(bracket_vort - les_vort)
        bracket_n = -(bracket_n - les_n)

    # Back to (x, batch/y, z), then C-order flattening expected by hw.cxx.
    out_vort = bracket_vort.permute(1, 0, 2).reshape(-1)
    out_n = bracket_n.permute(1, 0, 2).reshape(-1)
    iteration = np.asarray([0.0], dtype=np.float64)
    return np.concatenate(
        (iteration, out_vort.cpu().double().numpy(), out_n.cpu().double().numpy())
    )
