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
  PBOUT_UNET_DNS_INIT=1                  # full-resolution, inverse-only DNS mode
  PBOUT_UNET_DNS_INIT_DOWNSAMPLE=1       # downsample fine state and call U-Net
  PBOUT_UNET_RECONSTRUCT_CHANNELS=U,V,P   # U-Net channels; others stay incoming
  PBOUT_UNET_DEVICE=cuda                 # or cpu; default is automatic
  PBOUT_UNET_IMPLICIT=1                  # return -(DNS-LES) brackets
  PBOUT_UNET_INVERSE_CUTOFF_FRACTION=none
  PBOUT_UNET_INVERSE_CUTOFF_ORDER=8
  PBOUT_UNET_POST_CUTOFF_FRACTION=none
  PBOUT_UNET_POST_CUTOFF_ORDER=8

The cutoff variables override checkpoint values.  Set the fraction to ``none``
to disable it.
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
    filter_snapshot,
    germano_filter,
    inverse_germano_torch,
    load_dns_snapshot,
    load_checkpoint,
    spectral_cutoff_torch,
    validate_hw_checkpoint,
)
from visualization_common import (
    arakawa_bracket as shared_arakawa_bracket,
    prepare_run_output,
    save_bout_arakawa_terms,
    save_arakawa_closure_comparison,
    save_arakawa_comparison,
    save_dns_snapshot,
    save_initial_dns_fields,
)


_HERE = Path(__file__).resolve().parent
_DEFAULT_CHECKPOINT = (
    _HERE
    / "results/germano_training_output_v10/checkpoints/best.pt"
)
CHECKPOINT = Path(os.environ.get("PBOUT_UNET_CHECKPOINT", _DEFAULT_CHECKPOINT))
_dns_path = os.environ.get("PBOUT_UNET_DNS_PATH")
DNS_PATH = Path(_dns_path).expanduser() if _dns_path else None
DNS_INIT = os.environ.get("PBOUT_UNET_DNS_INIT", "0").lower() not in {
    "0", "false", "no", "off"
}
DNS_INIT_DOWNSAMPLE = os.environ.get(
    "PBOUT_UNET_DNS_INIT_DOWNSAMPLE", "0"
).lower() not in {"0", "false", "no", "off"}
if DNS_INIT_DOWNSAMPLE and not DNS_INIT:
    raise ValueError(
        "PBOUT_UNET_DNS_INIT_DOWNSAMPLE requires PBOUT_UNET_DNS_INIT=1"
    )
BOUT_INPUT = Path(
    os.environ.get("PBOUT_UNET_BOUT_INP", "data/BOUT.inp")
).expanduser()
IMPLICIT = os.environ.get("PBOUT_UNET_IMPLICIT", "1").lower() not in {
    "0", "false", "no", "off"
}
_BOUT_OUTPUT_TIMESTEP = None
_LAST_FIELD_OUTPUT_INDEX = -1


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
        "dns_init_downsample": DNS_INIT_DOWNSAMPLE,
        "initial_state": initial_state,
        "implicit": IMPLICIT,
        "vorticity_mode": VORTICITY_MODE,
        "dns_vorticity_mode": DNS_VORTICITY_MODE,
        "vorticity_operator": "learned_independent_channel",
        "field_names": FIELD_NAMES,
        "reconstruct_channels": RECONSTRUCT_CHANNELS,
        "model_output_channels": OUTPUT_CHANNELS,
        "fine_n": FINE_N,
        "coarse_n": COARSE_N,
        "lx": LX,
        "ly": LY,
        "alpha": ALPHA,
        "base_channels": BASE_CHANNELS,
        "inverse_cutoff_fraction": CUTOFF_FRACTION,
        "inverse_cutoff_order": CUTOFF_ORDER,
        "post_cutoff_fraction": POST_CUTOFF_FRACTION,
        "post_cutoff_order": POST_CUTOFF_ORDER,
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
    "PBOUT_UNET_RECONSTRUCT_CHANNELS", "all"
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
VORTICITY_MODE = "learned_channel"
DNS_VORTICITY_MODE = "independent_channel"
OUTPUT_CHANNELS = int(_checkpoint["model_output_channels"])
if FINE_N % COARSE_N:
    raise ValueError("Checkpoint fine_n must be divisible by coarse_n")

MODEL = None
if not DNS_INIT or DNS_INIT_DOWNSAMPLE:
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
    _checkpoint.get("inverse_cutoff_fraction"),
)
CUTOFF_ORDER = float(
    os.environ.get(
        "PBOUT_UNET_INVERSE_CUTOFF_ORDER",
        _checkpoint.get("inverse_cutoff_order", 8.0),
    )
)
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
def _reconstruct_dns(
    coarse: torch.Tensor,
    incoming_filtered_fine: torch.Tensor | None = None,
):
    """Return reconstructed DNS fields and the cutoff-filtered U-Net output."""
    if MODEL is None:
        raise RuntimeError(
            "U-Net reconstruction is disabled in inverse-only "
            "PBOUT_UNET_DNS_INIT mode"
        )
    normalised = (coarse - MEAN) / STD
    predicted = MODEL(normalised)
    filtered_fine = predicted * STD + MEAN
    inverse_input = spectral_cutoff_torch(
        filtered_fine, LX, LY, CUTOFF_FRACTION, CUTOFF_ORDER
    )
    if incoming_filtered_fine is not None:
        if incoming_filtered_fine.shape != inverse_input.shape:
            raise ValueError(
                "Incoming fine LES fields and U-Net reconstruction have "
                f"different shapes: {incoming_filtered_fine.shape} and "
                f"{inverse_input.shape}"
            )
        reconstruct_mask = torch.tensor(
            [name in RECONSTRUCT_CHANNELS for name in FIELD_NAMES],
            dtype=torch.bool,
            device=inverse_input.device,
        ).view(1, -1, 1, 1)
        inverse_input = torch.where(
            reconstruct_mask, inverse_input, incoming_filtered_fine
        )
    dns = inverse_germano_torch(inverse_input, ALPHA, LX, LY)
    dns = spectral_cutoff_torch(
        dns, LX, LY, POST_CUTOFF_FRACTION, POST_CUTOFF_ORDER
    )
    return dns, inverse_input


@torch.inference_mode()
def _inverse_germano_only(filtered_fields: torch.Tensor) -> torch.Tensor:
    """Reconstruct DNS fields directly, without the U-Net or grid resizing."""
    dns = inverse_germano_torch(filtered_fields, ALPHA, LX, LY)
    return dns


def _save_dns_init_unet_validation(
    time_step: int,
    simtime: float,
    incoming_filtered: torch.Tensor,
    coarse: torch.Tensor,
    predicted_filtered: torch.Tensor,
    reconstructed_dns: torch.Tensor,
) -> Path:
    """Save paired fine-grid references and U-Net results during integration."""
    reference_dns = _inverse_germano_only(incoming_filtered)
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
    """Return initial LES fields, optionally derived from a DNS snapshot.

    If ``PBOUT_UNET_DNS_INIT`` is enabled, ``PBOUT_UNET_DNS_PATH`` must point
    to a snapshot at the checkpoint's DNS resolution.  Its U and V fields are
    independently differentially filtered and returned at DNS resolution
    without calling the U-Net.

    If ``PBOUT_UNET_DNS_PATH`` is set, load the ``U`` (density) and ``V``
    (potential) arrays from that ``.npz`` DNS snapshot, apply the checkpoint's
    Germano filter, and spectrally resize them to the LES grid.  Otherwise,
    preserve the initial fields supplied by BOUT++.
    """
    output_dir = Path("results_U-Net")
    if output_dir.exists():
        shutil.rmtree(output_dir)

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
        reference_fields = _load_dns_uvp(DNS_PATH)
        fields = np.stack(
            [
                germano_filter(channel, ALPHA, LX, LY)
                for channel in reference_fields
            ]
        ).astype(NUMPY_DTYPE)

        fields_output, fields_spectrum_output = save_initial_dns_fields(
            fields,
            reference_fields,
            FIELD_NAMES,
            LX,
            LY,
            Path("results_U-Net") / "images",
            POST_CUTOFF_FRACTION,
        )
        arakawa_output, arakawa_spectrum_output = save_arakawa_comparison(
            fields,
            reference_fields,
            LX,
            LY,
            Path("results_U-Net") / "arakawa",
            stem="arakawa_DNS_initial",
            cutoff_fraction=POST_CUTOFF_FRACTION,
            spectrum_stem="arakawa_DNS_spectrum_initial.png",
        )

        integration_mode = (
            "incoming states will be downsampled and reconstructed by "
            "the U-Net during time integration;\n"
            if DNS_INIT_DOWNSAMPLE
            else "U-Net reconstruction is disabled;\n"
        )
        print(
            f"Initialized BOUT++ from filtered DNS snapshot {DNS_PATH};\n"
            f"resolution={FINE_N}, Lx={LX}, Ly={LY};\n"
            f"differential filter alpha={ALPHA:g}; fields were not resized "
            f"during initialization; {integration_mode}"
            f"saved filtered and original DNS fields to {fields_output};\n"
            f"saved their spectra to {fields_spectrum_output};\n"
            f"saved DNS Arakawa terms to {arakawa_output};\n"
            f"saved their spectra to {arakawa_spectrum_output}\n",
            flush=True,
        )

        print("\n\n", flush=True)
        print("------------------ U-Net parameters---------------", flush=True)
        initial_state = f"filtered DNS snapshot {DNS_PATH}"
        _print_init_metadata(initial_state)
        print("\n\n", flush=True)

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
    initial_dns, _ = _reconstruct_dns(coarse)
    initial_dns = initial_dns[0].detach().cpu().double().numpy()
    prepare_run_output(Path("results_U-Net"))

    fields_image_output, fields_spectrum_output = save_initial_dns_fields(
        initial_dns,
        reference_dns,
        FIELD_NAMES,
        LX,
        LY,
        Path("results_U-Net") / "images",
        POST_CUTOFF_FRACTION,
    )

    arakawa_image_output, arakawa_spectrum_output = save_arakawa_comparison(
        initial_dns,
        reference_dns,
        LX,
        LY,
        Path("results_U-Net") / "arakawa",
        stem="arakawa_DNS_initial",
        cutoff_fraction=POST_CUTOFF_FRACTION,
        spectrum_stem="arakawa_DNS_spectrum_initial.png",
    )

    print("\n\n", flush=True)
    print("------------------ U-Net parameters---------------", flush=True)
    _print_init_metadata(initial_state)
    print("\n\n", flush=True)

    return fields.reshape(-1).astype(np.float64, copy=False)


@torch.inference_mode()
def findLESTerms(pLES):
    values = np.asarray(pLES, dtype=np.float64).reshape(-1)
    if values.size < 4:
        raise ValueError("findLESTerms requires the four-value BOUT header")
    time_step = int(values[0])
    simtime = float(values[3])
    dx_les = float(values[2])
    simulation_n = FINE_N if DNS_INIT else COARSE_N
    _validate_les_spacing(dx_les, dx_les, simulation_n)
    required_channels = 5 if IMPLICIT else 3
    fields, batch = _decode_fields(
        values, 4, required_channels, simulation_n
    )

    simulation_fields = torch.from_numpy(fields[:, :3]).to(
        device=DEVICE, dtype=TORCH_DTYPE
    )
    coarse_fields = None
    if DNS_INIT_DOWNSAMPLE:
        coarse_fields = _spectral_resize_torch(
            simulation_fields, COARSE_N
        )
        dns, filtered_fine = _reconstruct_dns(
            coarse_fields, incoming_filtered_fine=simulation_fields
        )
    elif DNS_INIT:
        filtered_fine = simulation_fields
        dns = _inverse_germano_only(filtered_fine)
    else:
        dns, filtered_fine = _reconstruct_dns(simulation_fields)
    if _field_output_due(simtime):
        dns_np = dns.detach().cpu().double().numpy()
        save_dns_snapshot(
            dns_np,
            FIELD_NAMES,
            time_step,
            simtime,
            Path("results_U-Net"),
        )
        if DNS_INIT_DOWNSAMPLE:
            _save_dns_init_unet_validation(
                time_step,
                simtime,
                simulation_fields,
                coarse_fields,
                filtered_fine,
                dns,
            )

    dx_dns = LX / FINE_N
    dy_dns = LY / FINE_N
    phi = dns[:, 1:2]
    bracket_vort = _arakawa_bracket(
        phi, dns[:, 2:3], dx_dns, dy_dns
    )
    bracket_n = _arakawa_bracket(phi, dns[:, 0:1], dx_dns, dy_dns)

    dx_les_checkpoint = LX / simulation_n
    dy_les_checkpoint = LY / simulation_n
    les_phi = simulation_fields[:, 1:2]
    les_bracket_vort = _arakawa_bracket(
        les_phi,
        simulation_fields[:, 2:3],
        dx_les_checkpoint,
        dy_les_checkpoint,
    )[:, 0]
    les_bracket_n = _arakawa_bracket(
        les_phi,
        simulation_fields[:, 0:1],
        dx_les_checkpoint,
        dy_les_checkpoint,
    )[:, 0]

    # misc_styles.find_bracket filters the fine-grid bracket before returning it
    # to BOUT++.  Full-resolution DNS mode keeps that filtering step but does
    # not resize; standard U-Net mode returns terms on the checkpoint LES grid.
    bracket_vort = _germano_filter_torch(bracket_vort)
    bracket_n = _germano_filter_torch(bracket_n)
    if not DNS_INIT:
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

        if time_step % 100 == -1:
            save_arakawa_closure_comparison(
                time_step,
                simtime,
                (
                    filtered_vort.detach().cpu().double().numpy(),
                    filtered_n.detach().cpu().double().numpy(),
                ),
                (
                    les_vort.detach().cpu().double().numpy(),
                    les_n.detach().cpu().double().numpy(),
                ),
                (
                    les_bracket_vort.detach().cpu().double().numpy(),
                    les_bracket_n.detach().cpu().double().numpy(),
                ),
                (
                    bracket_vort.detach().cpu().double().numpy(),
                    bracket_n.detach().cpu().double().numpy(),
                ),
                Path(f"results_U-Net") / "arakawa",
            )

    # Back to (x, batch/y, z), then C-order flattening expected by hw.cxx.
    out_vort = bracket_vort.permute(1, 0, 2).reshape(-1)
    out_n = bracket_n.permute(1, 0, 2).reshape(-1)
    iteration = np.asarray([0.0], dtype=np.float64)
    return np.concatenate(
        (iteration, out_vort.cpu().double().numpy(), out_n.cpu().double().numpy())
    )
