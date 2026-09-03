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
  PBOUT_UNET_ALPHA=0.25                  # physical Germano filter width
  PBOUT_UNET_DNS_INIT=0                  # 0=filtered LES, 1=filtered DNS, 2=DNS
  PBOUT_UNET_RECONSTRUCT_CHANNELS=none    # mode-1 U-Net channels
  PBOUT_UNET_DEVICE=cuda                 # or cpu; default is automatic
  PBOUT_UNET_IMPLICIT=1                  # return -(DNS-LES) brackets
  PBOUT_UNET_WRITE_DIAGNOSTICS=2         # 0=off, 1=all, 2=fields only (default)
  PBOUT_UNET_OMEGA_FROM_POTENTIAL=1      # omega = Laplacian(phi) at DNS scale
  PBOUT_UNET_LAPLACIAN_ORDER=2            # periodic Laplacian order (2 or 4)
  PBOUT_UNET_INVERSE_CUTOFF_FRACTION=none
  PBOUT_UNET_INVERSE_CUTOFF_ORDER=8
  PBOUT_UNET_INVERSE_METHOD=exact          # exact or van_cittert
  PBOUT_UNET_VAN_CITTERT_ITERATIONS=1
  PBOUT_UNET_VAN_CITTERT_RELAXATION=1
  PBOUT_UNET_POST_CUTOFF_FRACTION=none
  PBOUT_UNET_POST_CUTOFF_ORDER=8
  PBOUT_UNET_POST_FILTER=none             # none/high_order/gaussian
The training-consistent inverse cutoff regularizes the U-Net output before
inverse Germano reconstruction. The optional post cutoff regularizes the
reconstructed DNS afterward.
Arakawa terms receive the Germano filter before Fourier downsampling.
"""

from __future__ import annotations

import os
import re
import site
import sys
import json
import shutil
import warnings

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

warnings.filterwarnings(
    "ignore",
    message=r"The value of the smallest subnormal.*is zero",
    category=UserWarning,
)

import numpy as np
import torch

from germano_common import (
    ResidualUNetSR,
    INVERSE_METHODS,
    POST_FILTER_TYPES,
    enforce_zero_spatial_mean,
    restore_channel_spatial_mean,
    germano_filter,
    reconstruct_germano_torch,
    load_checkpoint,
    periodic_laplacian_2nd_torch,
    periodic_laplacian_4th_torch,
    post_filter_torch,
    spectral_cutoff_torch,
    validate_hw_checkpoint,
    validate_grid_pair,
)
from visualization_common import (
    arakawa_bracket as shared_arakawa_bracket,
    prepare_run_output,
    save_bout_arakawa_terms,
    save_arakawa_closure_comparison,
    save_arakawa_comparison,
    save_arakawa_dns_field_comparison,
    save_dns_snapshot,
    save_energy_history,
    save_field_centerline_comparison,
    save_field_error_images,
    save_initial_dns_fields,
    save_png_animation,
    save_vorticity_comparison,
)


_HERE = Path(__file__).resolve().parent
_DEFAULT_CHECKPOINT = (
    _HERE
    / "results/germano_training_output_v10/checkpoints/best.pt"
)
CHECKPOINT = Path(os.environ.get("PBOUT_UNET_CHECKPOINT", _DEFAULT_CHECKPOINT))
_dns_path = os.environ.get("PBOUT_UNET_DNS_PATH")
DNS_PATH = Path(_dns_path).expanduser() if _dns_path else None

try:
    DNS_INIT = int(os.environ.get("PBOUT_UNET_DNS_INIT", "0"))
except ValueError as error:
    raise ValueError("PBOUT_UNET_DNS_INIT must be 0, 1, or 2") from error
if DNS_INIT not in (0, 1, 2):
    raise ValueError("PBOUT_UNET_DNS_INIT must be 0, 1, or 2")

# DNS_INIT is the single switch for the three requested behaviors.
BOUT_INPUT = Path(
    os.environ.get("PBOUT_UNET_BOUT_INP", "data/BOUT.inp")
).expanduser()
IMPLICIT = os.environ.get("PBOUT_UNET_IMPLICIT", "1").lower() not in {
    "0", "false", "no", "off"
}
try:
    WRITE_DIAGNOSTICS = int(
        os.environ.get("PBOUT_UNET_WRITE_DIAGNOSTICS", "2")
    )
except ValueError as error:
    raise ValueError(
        "PBOUT_UNET_WRITE_DIAGNOSTICS must be 0, 1, or 2"
    ) from error
if WRITE_DIAGNOSTICS not in (0, 1, 2):
    raise ValueError("PBOUT_UNET_WRITE_DIAGNOSTICS must be 0, 1, or 2")
OMEGA_FROM_POTENTIAL = os.environ.get(
    "PBOUT_UNET_OMEGA_FROM_POTENTIAL", "1"
).lower() not in {"0", "false", "no", "off"}
_BOUT_OUTPUT_TIMESTEP = None
_BOUT_NOUT = None
_LAST_FIELD_OUTPUT_INDEX = -1
_INITIAL_DNS_REFERENCE = None
_INITIAL_DNS_PLOT_LIMITS = None
_ENERGY_TIMES = []
_ENERGY_VALUES = []


def _optional_float(name: str, default):
    value = os.environ.get(name)
    if value is None:
        return default
    if value.lower() in {"none", "off", "disable", "disabled"}:
        return None
    return float(value)


def _bout_output_timestep() -> float:
    """Read the top-level BOUT++ physical-time output interval."""
    global _BOUT_OUTPUT_TIMESTEP
    if _BOUT_OUTPUT_TIMESTEP is not None:
        return _BOUT_OUTPUT_TIMESTEP
    if not BOUT_INPUT.is_file():
        raise FileNotFoundError(
            f"BOUT++ input file not found: {BOUT_INPUT}. Set "
            "PBOUT_UNET_BOUT_INP to the data/BOUT.inp used by this run."
        )

    assignment = re.compile(
        r"^\s*timestep\s*=\s*"
        r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
        r"\s*(?:#.*)?$",
        re.IGNORECASE,
    )
    in_top_level = True
    for line in BOUT_INPUT.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_top_level = False
        if not in_top_level:
            continue
        match = assignment.match(line)
        if match:
            timestep = float(match.group(1))
            if not np.isfinite(timestep) or timestep <= 0:
                raise ValueError(
                    f"{BOUT_INPUT}: timestep must be finite and positive"
                )
            _BOUT_OUTPUT_TIMESTEP = timestep
            return timestep
    raise ValueError(
        f"{BOUT_INPUT}: no top-level 'timestep = ...' setting was found"
    )


def _bout_nout() -> int:
    """Read the number of requested BOUT++ output intervals."""
    global _BOUT_NOUT
    if _BOUT_NOUT is not None:
        return _BOUT_NOUT
    assignment = re.compile(
        r"^\s*nout\s*=\s*(\d+)\s*(?:#.*)?$",
        re.IGNORECASE,
    )
    in_top_level = True
    for line in BOUT_INPUT.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            in_top_level = False
        if not in_top_level:
            continue
        match = assignment.match(line)
        if match:
            _BOUT_NOUT = int(match.group(1))
            if _BOUT_NOUT < 0:
                raise ValueError(f"{BOUT_INPUT}: nout must be non-negative")
            return _BOUT_NOUT
    raise ValueError(f"{BOUT_INPUT}: no top-level 'nout = ...' setting was found")


def _save_diagnostic_animations(output_dir: Path) -> None:
    """Animate every numbered PNG family after the final BOUT++ output."""
    images_dir = output_dir / "images"
    numbered_frame = re.compile(
        r"^(?P<prefix>.*?)(?P<step>\d+)"
        r"(?P<batch>_batch\d+)?\.png$"
    )
    image_families = {}
    for frame in images_dir.glob("*.png"):
        match = numbered_frame.match(frame.name)
        if match is None:
            continue
        prefix = match.group("prefix").rstrip("_")
        if prefix.endswith("time"):
            prefix = prefix[:-4].rstrip("_")
        batch = match.group("batch") or ""
        family = f"{prefix}{batch}"
        image_families.setdefault(family, []).append(
            (int(match.group("step")), frame)
        )

    animations = [
        (
            images_dir,
            [frame for _, frame in sorted(frames)],
            f"{family}_animation.gif",
        )
        for family, frames in sorted(image_families.items())
    ]
    animations.extend(
        (
            (
                output_dir / "arakawa",
                sorted(
                    (output_dir / "arakawa").glob(
                        "arakawa_DNS_time[0-9]*.png"
                    )
                ),
                "arakawa_DNS_terms_animation.gif",
            ),
            (
                output_dir / "arakawa",
                sorted(
                    (output_dir / "arakawa").glob(
                        "arakawa_DNS_spectrum_time[0-9]*.png"
                    )
                ),
                "arakawa_DNS_spectrum_animation.gif",
            ),
            (
                output_dir / "arakawa",
                sorted(
                    (output_dir / "arakawa").glob(
                        "arakawa_DNS_field_comparison_time[0-9]*.png"
                    )
                ),
                "arakawa_DNS_field_comparison_animation.gif",
            ),
        )
    )
    for directory, frames, name in animations:
        if frames:
            save_png_animation(frames, directory / name)
            print(
                f"Saved {len(frames)} diagnostic frames to "
                f"{directory / name}",
                flush=True,
            )


def _hw_energy(fields: torch.Tensor) -> torch.Tensor:
    """Return the StylES HW energy using fourth-order periodic gradients."""
    density = fields[:, 0]
    potential = fields[:, 1]
    dx = LX / potential.shape[-2]
    dy = LY / potential.shape[-1]
    dphi_dx = (
        -torch.roll(potential, shifts=-2, dims=-2)
        + 8.0 * torch.roll(potential, shifts=-1, dims=-2)
        - 8.0 * torch.roll(potential, shifts=1, dims=-2)
        + torch.roll(potential, shifts=2, dims=-2)
    ) / (12.0 * dx)
    dphi_dy = (
        -torch.roll(potential, shifts=-2, dims=-1)
        + 8.0 * torch.roll(potential, shifts=-1, dims=-1)
        - 8.0 * torch.roll(potential, shifts=1, dims=-1)
        + torch.roll(potential, shifts=2, dims=-1)
    ) / (12.0 * dy)
    density_energy = density.square()
    gradient_energy = dphi_dx.square() + dphi_dy.square()
    return 0.5 * LX * LY * (density_energy + gradient_energy).sum(
        dim=(-2, -1)
    )


def _field_output_due(simtime: float) -> bool:
    """Return true once at each BOUT++ physical-time output point."""
    global _LAST_FIELD_OUTPUT_INDEX
    timestep = _bout_output_timestep()
    output_index = int(round(simtime / timestep))
    output_time = output_index * timestep
    tolerance = 1e-8 * max(1.0, abs(simtime), abs(output_time))
    if (
        output_index < 0
        or abs(simtime - output_time) > tolerance
        or output_index <= _LAST_FIELD_OUTPUT_INDEX
    ):
        return False
    _LAST_FIELD_OUTPUT_INDEX = output_index
    return True


def _print_init_metadata(initial_state: str) -> None:
    """Print checkpoint metadata and the effective pBOUT configuration."""
    candidates = (
        CHECKPOINT.parent / "metadata.json",
        CHECKPOINT.parent.parent / "metadata.json",
    )
    metadata_path = next((path for path in candidates if path.is_file()), None)
    metadata = (
        json.loads(metadata_path.read_text())
        if metadata_path is not None
        else {}
    )
    if metadata_path is None:
        print(
            "Checkpoint metadata: metadata.json was not found beside the "
            "checkpoint directory."
        )
    else:
        print(f"Checkpoint metadata ({metadata_path}):")

    def print_values(value, prefix=""):
        if isinstance(value, dict):
            for name, nested_value in value.items():
                if not prefix and name in {
                    "tensorboard_dir",
                    "test_metrics",
                    "training_parameters",
                    "training_parameters_file",
                }:
                    continue
                nested_prefix = f"{prefix}.{name}" if prefix else name
                print_values(nested_value, nested_prefix)
        else:
            print(f"{prefix}: {json.dumps(value, default=str)}")

    print_values(metadata)
    effective_inference = {
        "checkpoint": str(CHECKPOINT),
        "checkpoint_epoch": _checkpoint.get("epoch"),
        "checkpoint_metadata_file": (
            str(metadata_path) if metadata_path is not None else None
        ),
        "device": str(DEVICE),
        "precision": PRECISION,
        "no_normalize": NO_NORMALIZE,
        "site_packages": _SITE_PACKAGES,
        "dns_path": str(DNS_PATH) if DNS_PATH is not None else None,
        "dns_init": DNS_INIT,
        "initial_state": initial_state,
        "implicit": IMPLICIT,
        "write_diagnostics": WRITE_DIAGNOSTICS,
        "vorticity_mode": VORTICITY_MODE,
        "dns_vorticity_mode": DNS_VORTICITY_MODE,
        "vorticity_operator": (
            "fourth_order_periodic_laplacian"
            if OMEGA_FROM_POTENTIAL and LAPLACIAN_ORDER == 4
            else (
                "second_order_periodic_laplacian"
                if OMEGA_FROM_POTENTIAL
                else "learned_independent_channel"
            )
        ),
        "omega_from_potential": OMEGA_FROM_POTENTIAL,
        "laplacian_order": LAPLACIAN_ORDER,
        "field_names": FIELD_NAMES,
        "reconstruct_channels": RECONSTRUCT_CHANNELS,
        "model_output_channels": OUTPUT_CHANNELS,
        "fine_n": FINE_N,
        "coarse_n": COARSE_N,
        "lx": LX,
        "ly": LY,
        "alpha": ALPHA,
        "base_channels": BASE_CHANNELS,
        "inverse_method": INVERSE_METHOD,
        "van_cittert_iterations": VAN_CITTERT_ITERATIONS,
        "van_cittert_relaxation": VAN_CITTERT_RELAXATION,
        "inverse_cutoff_fraction": CUTOFF_FRACTION,
        "inverse_cutoff_order": CUTOFF_ORDER,
        "post_cutoff_fraction": POST_CUTOFF_FRACTION,
        "post_cutoff_order": POST_CUTOFF_ORDER,
        "post_filter": POST_FILTER_TYPE,
    }
    print_values(effective_inference, "effective_inference")


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
NORMALIZATION_ENABLED = _checkpoint.get(
    "normalize",
    _checkpoint.get("training_parameters", {}).get("normalize", True),
)
NO_NORMALIZE = not NORMALIZATION_ENABLED
FIELD_NAMES = tuple(_checkpoint["field_names"])
_reconstruct_channels_value = os.environ.get(
    "PBOUT_UNET_RECONSTRUCT_CHANNELS", "none"
)
if _reconstruct_channels_value.strip().lower() == "all":
    RECONSTRUCT_CHANNELS = FIELD_NAMES
elif _reconstruct_channels_value.strip().lower() in {"none", "off"}:
    RECONSTRUCT_CHANNELS = ()
else:
    requested_channels = tuple(
        name.strip().upper()
        for name in _reconstruct_channels_value.split(",")
        if name.strip()
    )
    unknown_channels = sorted(set(requested_channels) - set(FIELD_NAMES))
    if unknown_channels:
        raise ValueError(
            "PBOUT_UNET_RECONSTRUCT_CHANNELS contains unknown channels "
            f"{unknown_channels}; choose from {FIELD_NAMES}, 'all', or 'none'"
        )
    RECONSTRUCT_CHANNELS = tuple(
        name for name in FIELD_NAMES if name in requested_channels
    )
FINE_N = int(_checkpoint["fine_n"])
COARSE_N = int(_checkpoint["coarse_n"])
LX = float(_checkpoint["lx"])
LY = float(_checkpoint["ly"])
_alpha_override = os.environ.get("PBOUT_UNET_ALPHA")
ALPHA = (
    float(_checkpoint["alpha"])
    if _alpha_override is None
    else float(_alpha_override)
)
if not np.isfinite(ALPHA) or ALPHA < 0:
    raise ValueError("PBOUT_UNET_ALPHA must be finite and non-negative")
BASE_CHANNELS = int(_checkpoint["base_channels"])
PRECISION = _checkpoint.get("precision", "float32")
if PRECISION not in ("float32", "float64"):
    raise ValueError(f"Unsupported checkpoint precision: {PRECISION}")
TORCH_DTYPE = getattr(torch, PRECISION)
NUMPY_DTYPE = getattr(np, PRECISION)
VORTICITY_MODE = _checkpoint["vorticity_mode"]
if VORTICITY_MODE == "laplacian_of_potential":
    OMEGA_FROM_POTENTIAL = True
DNS_VORTICITY_MODE = (
    "derived_from_potential"
    if OMEGA_FROM_POTENTIAL
    else "independent_channel"
)
_training_parameters = _checkpoint.get("training_parameters", {})
INVERSE_METHOD = os.environ.get(
    "PBOUT_UNET_INVERSE_METHOD",
    _checkpoint.get(
        "inverse_method", _training_parameters.get("inverse_method", "exact")
    ),
).lower().replace("-", "_")
if INVERSE_METHOD not in INVERSE_METHODS:
    raise ValueError(
        "PBOUT_UNET_INVERSE_METHOD must be one of "
        f"{', '.join(INVERSE_METHODS)}."
    )
VAN_CITTERT_ITERATIONS = int(
    os.environ.get(
        "PBOUT_UNET_VAN_CITTERT_ITERATIONS",
        _checkpoint.get(
            "van_cittert_iterations",
            _training_parameters.get("van_cittert_iterations", 1),
        ),
    )
)
VAN_CITTERT_RELAXATION = float(
    os.environ.get(
        "PBOUT_UNET_VAN_CITTERT_RELAXATION",
        _checkpoint.get(
            "van_cittert_relaxation",
            _training_parameters.get("van_cittert_relaxation", 1.0),
        ),
    )
)
if VAN_CITTERT_ITERATIONS < 0:
    raise ValueError("PBOUT_UNET_VAN_CITTERT_ITERATIONS must be non-negative.")
if not 0 < VAN_CITTERT_RELAXATION <= 1:
    raise ValueError("PBOUT_UNET_VAN_CITTERT_RELAXATION must be in (0, 1].")
LAPLACIAN_ORDER = int(
    os.environ.get(
        "PBOUT_UNET_LAPLACIAN_ORDER",
        _checkpoint.get(
            "laplacian_order",
            _training_parameters.get("laplacian_order", 2),
        ),
    )
)
if LAPLACIAN_ORDER not in (2, 4):
    raise ValueError("PBOUT_UNET_LAPLACIAN_ORDER must be 2 or 4")
OUTPUT_CHANNELS = int(_checkpoint["model_output_channels"])
SCALE_FACTOR = validate_grid_pair(FINE_N, COARSE_N)

MODEL = ResidualUNetSR(
    len(FIELD_NAMES), BASE_CHANNELS, SCALE_FACTOR, OUTPUT_CHANNELS
).to(device=DEVICE, dtype=TORCH_DTYPE)
MODEL.load_state_dict(_checkpoint["model"])
MODEL.eval()
MODEL.requires_grad_(False)

MEAN = torch.as_tensor(_checkpoint["mean"], dtype=TORCH_DTYPE, device=DEVICE)
STD = torch.as_tensor(_checkpoint["std"], dtype=TORCH_DTYPE, device=DEVICE)
CUTOFF_FRACTION = _optional_float(
    "PBOUT_UNET_INVERSE_CUTOFF_FRACTION",
    _checkpoint.get("inverse_cutoff_fraction"),
)
CUTOFF_ORDER = float(
    os.environ.get(
        "PBOUT_UNET_INVERSE_CUTOFF_ORDER",
        _checkpoint.get("inverse_cutoff_order", 8.0),
    )
)
if CUTOFF_FRACTION is not None and not 0 < CUTOFF_FRACTION <= 1:
    raise ValueError(
        "PBOUT_UNET_INVERSE_CUTOFF_FRACTION must be in (0, 1]."
    )
if not np.isfinite(CUTOFF_ORDER) or CUTOFF_ORDER <= 0:
    raise ValueError("PBOUT_UNET_INVERSE_CUTOFF_ORDER must be positive.")
POST_CUTOFF_FRACTION = _optional_float(
    "PBOUT_UNET_POST_CUTOFF_FRACTION",
    _checkpoint.get("post_cutoff_fraction"),
)
POST_CUTOFF_ORDER = float(
    os.environ.get(
        "PBOUT_UNET_POST_CUTOFF_ORDER",
        _checkpoint.get("post_cutoff_order", 8.0),
    )
)
if (
    POST_CUTOFF_FRACTION is not None
    and not 0 < POST_CUTOFF_FRACTION <= 1
):
    raise ValueError(
        "PBOUT_UNET_POST_CUTOFF_FRACTION must be in (0, 1]."
    )
if not np.isfinite(POST_CUTOFF_ORDER) or POST_CUTOFF_ORDER <= 0:
    raise ValueError("PBOUT_UNET_POST_CUTOFF_ORDER must be positive.")
_checkpoint_post_filter = _checkpoint.get(
    "post_filter",
    _training_parameters.get("post_filter", "none"),
)
if _checkpoint_post_filter == "smooth_exponential_cutoff":
    _checkpoint_post_filter = "high_order"
POST_FILTER_TYPE = os.environ.get(
    "PBOUT_UNET_POST_FILTER", _checkpoint_post_filter
).lower().replace("-", "_")
if POST_FILTER_TYPE == "dns_spectrum":
    print(
        "Checkpoint requested the removed dns_spectrum post-filter; "
        "using no post-filter.",
        flush=True,
    )
    POST_FILTER_TYPE = "none"
if POST_FILTER_TYPE not in POST_FILTER_TYPES:
    raise ValueError(
        "PBOUT_UNET_POST_FILTER must be one of "
        f"{', '.join(POST_FILTER_TYPES)}."
    )
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


def _arakawa_terms_torch(dns: torch.Tensor):
    """Calculate the two fine-grid Arakawa terms used by BOUT++."""
    dx_dns = LX / dns.shape[-2]
    dy_dns = LY / dns.shape[-1]
    phi = dns[:, 1:2]
    return (
        _arakawa_bracket(phi, dns[:, 2:3], dx_dns, dy_dns),
        _arakawa_bracket(phi, dns[:, 0:1], dx_dns, dy_dns),
    )


@torch.inference_mode()
def _arakawa_spectrum_stages_numpy(
    dns_fields: np.ndarray,
) -> list[tuple[str, tuple[np.ndarray, np.ndarray], str]]:
    """Calculate Germano-filter and downsample diagnostic stages."""
    dns = torch.as_tensor(
        dns_fields[None], device=DEVICE, dtype=TORCH_DTYPE
    )
    terms = _arakawa_terms_torch(dns)
    germano = tuple(_germano_filter_torch(term) for term in terms)
    downsampled = tuple(
        _spectral_resize_torch(term, COARSE_N) for term in germano
    )

    def numpy_terms(values):
        return tuple(
            term[0, 0].detach().cpu().double().numpy() for term in values
        )

    return [
        ("After Germano filter", numpy_terms(germano), "tab:green"),
        ("After downsampling", numpy_terms(downsampled), "tab:red"),
    ]


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


def _decode_fields(
    values: np.ndarray,
    offset: int,
    required_channels: int,
    grid_n: int = COARSE_N,
):
    payload = values[offset:]
    plane = grid_n * grid_n
    # The current hw.cxx always allocates five blocks, even in explicit mode;
    # also accept a future compact three-block explicit payload.
    stored_channels = 5 if payload.size % (5 * plane) == 0 else 3
    if (
        stored_channels < required_channels
        or payload.size % (stored_channels * plane)
    ):
        raise ValueError(
            f"BOUT payload has {payload.size} values; expected a multiple of "
            f"{required_channels * plane} for a {grid_n}x{grid_n} grid"
        )
    batch = payload.size // (stored_channels * plane)
    arrays = []
    for channel in range(required_channels):
        block = payload[
            channel * batch * plane:(channel + 1) * batch * plane
        ]
        # C++ block order is (x, batch/y, z).
        arrays.append(block.reshape(grid_n, batch, grid_n).transpose(1, 0, 2))
    return np.stack(arrays, axis=1), batch


@torch.inference_mode()
def _vorticity_from_potential(dns: torch.Tensor) -> torch.Tensor:
    """Calculate the configured periodic Laplacian of reconstructed phi."""
    laplacian = {
        2: periodic_laplacian_2nd_torch,
        4: periodic_laplacian_4th_torch,
    }[LAPLACIAN_ORDER]
    potential_index = FIELD_NAMES.index("V")
    return laplacian(
        dns[:, potential_index:potential_index + 1], LX, LY
    )


@torch.inference_mode()
def _set_omega_from_potential(dns: torch.Tensor) -> torch.Tensor:
    """Replace DNS omega with the periodic Laplacian of reconstructed phi."""
    if not OMEGA_FROM_POTENTIAL:
        return dns
    omega_index = FIELD_NAMES.index("P")
    omega = _vorticity_from_potential(dns)
    return torch.cat(
        (
            dns[:, :omega_index],
            omega,
            dns[:, omega_index + 1:],
        ),
        dim=1,
    )


@torch.inference_mode()
def _restore_vorticity_mean(
    dns: torch.Tensor, source_fields: torch.Tensor
) -> torch.Tensor:
    """Restore the vorticity zero mode carried by the BOUT++ state.

    A periodic Laplacian inversion only determines the mean-free vorticity.
    BOUT++ evolves the vorticity zero mode separately, so retain its incoming
    spatial mean after reconstructing vorticity from the potential.
    """
    return restore_channel_spatial_mean(
        dns, source_fields, FIELD_NAMES.index("P")
    )


@torch.inference_mode()
def _append_vorticity_for_diagnostics(fields: torch.Tensor) -> torch.Tensor:
    """Return U,V,P diagnostics for a two-output U-Net field."""
    if fields.shape[1] == len(FIELD_NAMES):
        return fields
    if fields.shape[1] != 2:
        raise ValueError(f"Expected two or three channels, got {fields.shape[1]}")
    return torch.cat((fields, _vorticity_from_potential(fields)), dim=1)


@torch.inference_mode()
def _reconstruct_dns(
    coarse: torch.Tensor,
    incoming_filtered_fine: torch.Tensor | None = None,
):
    """Return inverse-Germano DNS, replacing only selected inputs by U-Net."""
    normalised = (coarse - MEAN) / STD
    predicted = MODEL(normalised)
    unet_output = (
        predicted * STD[:, :OUTPUT_CHANNELS]
        + MEAN[:, :OUTPUT_CHANNELS]
    )
    inverse_input = unet_output
    if incoming_filtered_fine is not None:
        incoming_filtered_fine = incoming_filtered_fine[:, :OUTPUT_CHANNELS]
        if incoming_filtered_fine.shape != inverse_input.shape:
            raise ValueError(
                "Incoming fine LES fields and U-Net reconstruction have "
                f"different shapes: {incoming_filtered_fine.shape} and "
                f"{inverse_input.shape}"
            )
        reconstruct_mask = torch.tensor(
            [
                name in RECONSTRUCT_CHANNELS
                for name in FIELD_NAMES[:OUTPUT_CHANNELS]
            ],
            dtype=torch.bool,
            device=inverse_input.device,
        ).view(1, -1, 1, 1)
        inverse_input = torch.where(
            reconstruct_mask, inverse_input, incoming_filtered_fine
        )
    inverse_input = spectral_cutoff_torch(
        inverse_input,
        LX,
        LY,
        CUTOFF_FRACTION,
        CUTOFF_ORDER,
    )
    dns = reconstruct_germano_torch(
        inverse_input,
        ALPHA,
        LX,
        LY,
        INVERSE_METHOD,
        VAN_CITTERT_ITERATIONS,
        VAN_CITTERT_RELAXATION,
    )
    dns = post_filter_torch(
        dns,
        LX,
        LY,
        POST_FILTER_TYPE,
        POST_CUTOFF_FRACTION,
        POST_CUTOFF_ORDER,
    )
    reconstructed_vorticity = (
        _vorticity_from_potential(dns)
        if VORTICITY_MODE == "laplacian_of_potential"
        else dns[:, FIELD_NAMES.index("P"):][:, :1]
    )
    # Derive omega last from the reconstructed potential. This avoids
    # independently inverse-reconstructing omega.
    dns = _set_omega_from_potential(dns)
    dns = enforce_zero_spatial_mean(dns)
    dns = _restore_vorticity_mean(dns, coarse)
    reconstructed_vorticity = dns[:, FIELD_NAMES.index("P"):][:, :1]
    return (
        dns,
        _append_vorticity_for_diagnostics(unet_output),
        _append_vorticity_for_diagnostics(inverse_input),
        reconstructed_vorticity,
    )


@torch.inference_mode()
def _inverse_germano_only(
    filtered_fields: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Reconstruct DNS and independent vorticity without U-Net or resizing."""
    inverse_input = spectral_cutoff_torch(
        filtered_fields,
        LX,
        LY,
        CUTOFF_FRACTION,
        CUTOFF_ORDER,
    )
    dns = reconstruct_germano_torch(
        inverse_input,
        ALPHA,
        LX,
        LY,
        INVERSE_METHOD,
        VAN_CITTERT_ITERATIONS,
        VAN_CITTERT_RELAXATION,
    )
    dns = post_filter_torch(
        dns,
        LX,
        LY,
        POST_FILTER_TYPE,
        POST_CUTOFF_FRACTION,
        POST_CUTOFF_ORDER,
    )
    reconstructed_vorticity = dns[:, FIELD_NAMES.index("P"):][:, :1]
    dns = _set_omega_from_potential(dns)
    dns = enforce_zero_spatial_mean(dns)
    dns = _restore_vorticity_mean(dns, filtered_fields)
    reconstructed_vorticity = dns[:, FIELD_NAMES.index("P"):][:, :1]
    return (
        dns,
        inverse_input,
        reconstructed_vorticity,
    )


def _save_dns_init_unet_validation(
    time_step: int,
    simtime: float,
    incoming_filtered: torch.Tensor,
    coarse: torch.Tensor,
    predicted_filtered: torch.Tensor,
    reconstructed_dns: torch.Tensor,
) -> Path:
    """Save paired fine-grid references and U-Net results during integration."""
    reference_dns, _, _ = _inverse_germano_only(incoming_filtered)
    arrays = {
        "incoming_filtered_fine": incoming_filtered,
        "downsampled_coarse": coarse,
        "unet_filtered_fine": predicted_filtered,
        "inverse_only_reference_dns": reference_dns,
        "unet_reconstructed_dns": reconstructed_dns,
    }
    numpy_arrays = {
        name: value.detach().cpu().double().numpy()
        for name, value in arrays.items()
    }

    def relative_l2(reference, prediction):
        axes = tuple(range(2, reference.ndim))
        numerator = np.sqrt(np.sum((prediction - reference) ** 2, axis=axes))
        denominator = np.sqrt(np.sum(reference**2, axis=axes))
        return numerator / np.maximum(denominator, np.finfo(np.float64).tiny)

    output_dir = Path("results_U-Net") / "validation"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"unet_validation_{time_step:07d}.npz"
    np.savez_compressed(
        output_path,
        pStep=np.asarray(time_step),
        simtime=np.asarray(simtime),
        lx=np.asarray(LX),
        ly=np.asarray(LY),
        dns_vorticity_mode=np.asarray(DNS_VORTICITY_MODE),
        filtered_relative_l2=relative_l2(
            numpy_arrays["incoming_filtered_fine"],
            numpy_arrays["unet_filtered_fine"],
        ),
        dns_relative_l2=relative_l2(
            numpy_arrays["inverse_only_reference_dns"],
            numpy_arrays["unet_reconstructed_dns"],
        ),
        **numpy_arrays,
    )
    return output_path


def _validate_les_spacing(
    dx: float, dy: float, grid_n: int = COARSE_N
) -> None:
    """Ensure BOUT++ and the trained checkpoint describe the same domain."""
    bout_lx = dx * grid_n
    bout_ly = dy * grid_n
    if not np.isclose(bout_lx, LX, rtol=1e-10, atol=1e-12) or not np.isclose(
        bout_ly, LY, rtol=1e-10, atol=1e-12
    ):
        raise ValueError(
            "BOUT++ domain does not match the U-Net training domain: "
            f"BOUT++ Lx={bout_lx}, Ly={bout_ly}; checkpoint Lx={LX}, Ly={LY}."
        )


def saveArakawaTerms(pArakawa):
    """BOUT++ callback delegating Arakawa output to visualization helpers."""
    if WRITE_DIAGNOSTICS == 1:
        save_bout_arakawa_terms(pArakawa, Path("results_U-Net_DNS"))
    return None


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
    """Return initial fields according to DNS_INIT.

    DNS_INIT=0:
        initialize from the reference DNS field, apply the Germano filter,
        and downsample to the coarse LES grid.
    DNS_INIT=1:
        initialize from the reference DNS field, apply the Germano filter
        and use the fine DNS grid when no reconstruction channels are selected.
        With selected channels, downsample BOUT++ to the coarse grid and use
        the U-Net to reconstruct a fine-grid inverse-Germano DNS for Arakawa.
    DNS_INIT=2:
        initialize from the reference DNS field, do not apply filtering
        or downsampling, and use the DNS fields directly.
    """
    global _INITIAL_DNS_REFERENCE, _INITIAL_DNS_PLOT_LIMITS
    global _ENERGY_TIMES, _ENERGY_VALUES
    global _LAST_FIELD_OUTPUT_INDEX

    _INITIAL_DNS_REFERENCE = None
    _ENERGY_TIMES = []
    _ENERGY_VALUES = []
    _LAST_FIELD_OUTPUT_INDEX = -1
    output_dir = Path("results_U-Net")
    if output_dir.exists():
        shutil.rmtree(output_dir)

    if DNS_PATH is None:
        raise ValueError(
            "PBOUT_UNET_DNS_PATH is required for PBOUT_UNET_DNS_INIT "
            "modes 0, 1, and 2."
        )
    if not DNS_PATH.is_file():
        raise FileNotFoundError(
            f"DNS snapshot not found: {DNS_PATH}. "
            "Set PBOUT_UNET_DNS_PATH to an existing .npz file."
        )

    simulation_fields = _load_dns_uvp(DNS_PATH)
    _INITIAL_DNS_REFERENCE = simulation_fields.copy()

    if DNS_INIT in (0, 1):
        source_fields = np.stack(
            [
                germano_filter(channel, ALPHA, LX, LY)
                for channel in simulation_fields
            ]
        ).astype(NUMPY_DTYPE)
    else:
        source_fields = simulation_fields

    source_torch = torch.from_numpy(source_fields[None]).to(
        device=DEVICE, dtype=TORCH_DTYPE
    )
    if DNS_INIT == 0:
        coarse_torch = _spectral_resize_torch(source_torch, COARSE_N)
        fields = coarse_torch[0].detach().cpu().numpy().astype(
            NUMPY_DTYPE, copy=False
        )
        initial_state = "Germano-filtered and downsampled DNS snapshot"
    elif DNS_INIT == 1:
        if RECONSTRUCT_CHANNELS:
            coarse_torch = _spectral_resize_torch(source_torch, COARSE_N)
            fields = coarse_torch[0].detach().cpu().numpy().astype(
                NUMPY_DTYPE, copy=False
            )
            initial_state = (
                "Germano-filtered and downsampled DNS snapshot for "
                "selected-channel reconstruction"
            )
        else:
            fields = source_fields
            initial_state = "Germano-filtered fine-grid DNS snapshot"
    else:
        fields = source_fields
        initial_state = "original unfiltered DNS snapshot"

    with torch.inference_mode():
        if DNS_INIT == 0:
            initial_dns, _, _, initial_reconstructed_vorticity = _reconstruct_dns(
                coarse_torch
            )
            diagnostic_mode = "downsampled and U-Net reconstructed"
        elif DNS_INIT == 1:
            if RECONSTRUCT_CHANNELS:
                (
                    initial_dns,
                    _,
                    _,
                    initial_reconstructed_vorticity,
                ) = _reconstruct_dns(
                    coarse_torch,
                    incoming_filtered_fine=source_torch,
                )
                diagnostic_mode = (
                    "fine-grid filtered and U-Net/inverse-Germano reconstructed"
                )
            else:
                (
                    initial_dns,
                    _,
                    initial_reconstructed_vorticity,
                ) = _inverse_germano_only(source_torch)
                diagnostic_mode = (
                    "fine-grid filtered and inverse-Germano reconstructed"
                )
        else:
            initial_dns = source_torch
            initial_reconstructed_vorticity = _vorticity_from_potential(
                initial_dns
            )
            diagnostic_mode = (
                "fine-grid Germano-filtered fields"
                if DNS_INIT == 1
                else "original unfiltered DNS fields"
            )

    initial_laplacian_vorticity = _vorticity_from_potential(initial_dns)
    initial_dns_np = initial_dns[0].detach().cpu().double().numpy()
    _INITIAL_DNS_PLOT_LIMITS = tuple(
        (float(np.min(field)), float(np.max(field)))
        for field in initial_dns_np
    )
    if WRITE_DIAGNOSTICS == 1:
        prepare_run_output(Path("results_U-Net"))

        fields_image_output, fields_spectrum_output = save_initial_dns_fields(
            initial_dns_np,
            simulation_fields,
            FIELD_NAMES,
            LX,
            LY,
            Path("results_U-Net") / "images",
            None,
        )
        save_vorticity_comparison(
            initial_reconstructed_vorticity[0, 0].detach().cpu().double().numpy(),
            initial_laplacian_vorticity[0, 0].detach().cpu().double().numpy(),
            Path("results_U-Net") / "images/vorticity_comparison_initial.png",
            "Initial reconstructed and Laplacian-derived vorticity",
        )

        arakawa_image_output, arakawa_spectrum_output = save_arakawa_comparison(
            initial_dns_np,
            simulation_fields,
            LX,
            LY,
            Path("results_U-Net") / "arakawa",
            stem="arakawa_DNS_initial",
            cutoff_fraction=None,
            spectrum_stem="arakawa_DNS_spectrum_initial.png",
            reference_label="Reference DNS Arakawa",
            current_terms_label=(
                "Reconstructed DNS Arakawa"
                if DNS_INIT in (0, 1)
                else (
                    "Filtered DNS Arakawa"
                    if DNS_INIT == 1
                    else "DNS Arakawa"
                )
            ),
            spectrum_variants=_arakawa_spectrum_stages_numpy(initial_dns_np),
        )

    print("\n\n", flush=True)
    print("------------------ U-Net parameters---------------", flush=True)
    _print_init_metadata(initial_state)
    print("\n\n", flush=True)

    print(
        f"Initialized BOUT++ from {initial_state} {DNS_PATH};\n"
        f"resolution={fields.shape[-1]}, Lx={LX}, Ly={LY};\n"
        f"Germano filter="
        f"{'enabled' if DNS_INIT in (0, 1) else 'disabled'}"
        f", alpha={ALPHA:g}; diagnostic mode={diagnostic_mode};\n"
        + (
            f"saved fields to {fields_image_output} and spectra to "
            f"{fields_spectrum_output};\n"
            f"saved DNS Arakawa terms to {arakawa_image_output} and spectra "
            f"to {arakawa_spectrum_output}\n"
            if WRITE_DIAGNOSTICS == 1
            else (
                "saving field diagnostics only\n"
                if WRITE_DIAGNOSTICS == 2
                else "image and field diagnostics disabled\n"
            )
        ),
        flush=True,
    )

    return fields.reshape(-1).astype(np.float64, copy=False)


@torch.inference_mode()
def findLESTerms(pLES):
    values = np.asarray(pLES, dtype=np.float64).reshape(-1)
    if values.size < 4:
        raise ValueError("findLESTerms requires the four-value BOUT header")

    time_step = int(values[0])
    simtime = float(values[3])
    dx_les = float(values[2])

    simulation_n = (
        FINE_N
        if DNS_INIT == 2 or (DNS_INIT == 1 and not RECONSTRUCT_CHANNELS)
        else COARSE_N
    )
    _validate_les_spacing(dx_les, dx_les, simulation_n)

    required_channels = 5 if IMPLICIT else 3
    fields, _ = _decode_fields(values, 4, required_channels, simulation_n)

    simulation_fields = torch.from_numpy(fields[:, :3]).to(
        device=DEVICE, dtype=TORCH_DTYPE
    )

    coarse_fields = None
    if DNS_INIT == 0:
        coarse_fields = simulation_fields
        (
            dns,
            unet_output,
            inverse_filtered_output,
            reconstructed_vorticity,
        ) = _reconstruct_dns(coarse_fields)
    elif DNS_INIT == 1:
        if RECONSTRUCT_CHANNELS:
            coarse_fields = simulation_fields
            incoming_filtered_fine = _spectral_resize_torch(
                simulation_fields, FINE_N
            )
            (
                dns,
                unet_output,
                inverse_filtered_output,
                reconstructed_vorticity,
            ) = _reconstruct_dns(
                coarse_fields,
                incoming_filtered_fine=incoming_filtered_fine,
            )
        else:
            (
                dns,
                inverse_filtered_output,
                reconstructed_vorticity,
            ) = _inverse_germano_only(simulation_fields)
            unet_output = simulation_fields
    else:
        dns = simulation_fields
        unet_output = simulation_fields
        inverse_filtered_output = simulation_fields
        reconstructed_vorticity = _vorticity_from_potential(dns)

    diagnostics_due = WRITE_DIAGNOSTICS and _field_output_due(simtime)
    simulation_fields_np = None
    if diagnostics_due:
        dns_np = dns.detach().cpu().double().numpy()
        les_input_np = (
            coarse_fields if coarse_fields is not None else simulation_fields
        ).detach().cpu().double().numpy()
        unet_output_np = unet_output.detach().cpu().double().numpy()
        inverse_filtered_output_np = (
            inverse_filtered_output.detach().cpu().double().numpy()
        )

        if simulation_n == FINE_N:
            simulation_fields_np = simulation_fields.detach().cpu().double().numpy()
        elif _INITIAL_DNS_REFERENCE is not None:
            simulation_fields_np = np.broadcast_to(
                _INITIAL_DNS_REFERENCE[None],
                (dns_np.shape[0],) + _INITIAL_DNS_REFERENCE.shape,
            )

        save_dns_snapshot(
            dns_np,
            FIELD_NAMES,
            time_step,
            simtime,
            Path("results_U-Net"),
            save_images=WRITE_DIAGNOSTICS == 1,
            reference_dns=simulation_fields_np,
            lx=LX,
            ly=LY,
            cutoff_fraction=None,
            les_input=les_input_np,
            unet_output=unet_output_np,
            inverse_filtered_output=inverse_filtered_output_np,
            reference_label=(
                "Incoming BOUT++ Germano-filtered DNS"
                if DNS_INIT == 1
                else "Incoming BOUT++ DNS"
            ),
            field_limits=_INITIAL_DNS_PLOT_LIMITS,
        )

        if WRITE_DIAGNOSTICS == 1:
            _ENERGY_TIMES.append(simtime)
            _ENERGY_VALUES.append(
                _hw_energy(dns).detach().cpu().double().numpy()
            )

        if WRITE_DIAGNOSTICS == 1:
            laplacian_vorticity = _vorticity_from_potential(dns)
            reconstructed_vorticity_np = (
                reconstructed_vorticity.detach().cpu().double().numpy()
            )
            laplacian_vorticity_np = (
                laplacian_vorticity.detach().cpu().double().numpy()
            )
            centerline_dns_np = dns_np.copy()
            vorticity_index = FIELD_NAMES.index("P")
            centerline_dns_np[:, vorticity_index] = reconstructed_vorticity_np[:, 0]

            images_dir = Path("results_U-Net") / "images"
            for batch_index in range(dns.shape[0]):
                suffix = "" if dns.shape[0] == 1 else f"_batch{batch_index:03d}"
                save_vorticity_comparison(
                    reconstructed_vorticity_np[batch_index, 0],
                    laplacian_vorticity_np[batch_index, 0],
                    images_dir / f"vorticity_comparison_time{time_step:07d}{suffix}.png",
                    (
                        "Reconstructed and Laplacian-derived vorticity, "
                        f"step {time_step}, t={simtime:g}"
                    ),
                )

                if simulation_fields_np is not None:
                    save_field_error_images(
                        dns_np[batch_index],
                        simulation_fields_np[batch_index],
                        FIELD_NAMES,
                        images_dir / (
                            "fields_reconstruction_error_time"
                            f"{time_step:07d}{suffix}.png"
                        ),
                        title_prefix="Reconstructed − incoming BOUT++ DNS",
                    )
                    save_field_centerline_comparison(
                        centerline_dns_np[batch_index],
                        simulation_fields_np[batch_index],
                        FIELD_NAMES,
                        images_dir / (
                            "fields_reconstruction_centerline_time"
                            f"{time_step:07d}{suffix}.png"
                        ),
                        reconstructed_label="Inverse-Germano reconstructed DNS",
                        reference_label=(
                            "Incoming BOUT++ Germano-filtered DNS"
                            if DNS_INIT == 1
                            else "Incoming BOUT++ DNS"
                        ),
                        additional_channel_lines={
                            vorticity_index: (
                                laplacian_vorticity_np[batch_index, 0],
                                r"$\omega=\nabla^2\phi$",
                            )
                        },
                    )

    arakawa_fields = simulation_fields if DNS_INIT == 2 else dns
    dx_dns = LX / arakawa_fields.shape[-2]
    dy_dns = LY / arakawa_fields.shape[-1]

    phi = arakawa_fields[:, 1:2]
    bracket_vort = _arakawa_bracket(phi, arakawa_fields[:, 2:3], dx_dns, dy_dns)
    bracket_n = _arakawa_bracket(phi, arakawa_fields[:, 0:1], dx_dns, dy_dns)

    # Filtering is set by the physical Germano width, independently of the
    # grid on which BOUT++ evolves the fields.  In DNS_INIT=1 with no selected
    # reconstruction channels, the filtered Arakawa terms stay on the fine
    # grid; selecting channels requires returning them on the coarse grid.
    if DNS_INIT in (0, 1) and ALPHA > 0:
        bracket_vort = _germano_filter_torch(bracket_vort)
        bracket_n = _germano_filter_torch(bracket_n)

    if DNS_INIT == 0 or (DNS_INIT == 1 and RECONSTRUCT_CHANNELS):
        bracket_vort = _spectral_resize_torch(bracket_vort, COARSE_N)
        bracket_n = _spectral_resize_torch(bracket_n, COARSE_N)

    bracket_vort = bracket_vort[:, 0]
    bracket_n = bracket_n[:, 0]

    if IMPLICIT:
        les_vort = torch.from_numpy(fields[:, 3]).to(
            device=DEVICE, dtype=TORCH_DTYPE
        )
        les_n = torch.from_numpy(fields[:, 4]).to(
            device=DEVICE, dtype=TORCH_DTYPE
        )

        filtered_vort = bracket_vort
        filtered_n = bracket_n
        bracket_vort = les_vort - filtered_vort
        bracket_n = les_n - filtered_n

    out_vort = bracket_vort.permute(1, 0, 2).reshape(-1)
    out_n = bracket_n.permute(1, 0, 2).reshape(-1)
    if (
        WRITE_DIAGNOSTICS == 1
        and diagnostics_due
        and _LAST_FIELD_OUTPUT_INDEX >= _bout_nout()
    ):
        save_energy_history(
            np.asarray(_ENERGY_TIMES, dtype=np.float64),
            np.asarray(_ENERGY_VALUES, dtype=np.float64),
            Path("results_U-Net") / "energy",
        )
        _save_diagnostic_animations(Path("results_U-Net"))
    iteration = np.asarray([0.0], dtype=np.float64)
    return np.concatenate(
        (iteration, out_vort.cpu().double().numpy(), out_n.cpu().double().numpy())
    )
