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
  PBOUT_UNET_DEVICE=cuda                 # or cpu; default is automatic
  PBOUT_UNET_IMPLICIT=1                  # return -(DNS-LES) brackets
  PBOUT_UNET_VORTICITY_FROM_POTENTIAL=1  # enforce vort = Laplacian(phi)
  PBOUT_UNET_INVERSE_CUTOFF_FRACTION=0.75
  PBOUT_UNET_INVERSE_CUTOFF_ORDER=8

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

from germano_common import (
    ResidualUNetSR,
    filter_snapshot,
    hw_fields_torch,
    inverse_germano_torch,
    load_dns_snapshot,
    load_checkpoint,
    periodic_laplacian_4th_torch,
    spectral_cutoff_torch,
    validate_hw_checkpoint,
)


_HERE = Path(__file__).resolve().parent
_DEFAULT_CHECKPOINT = (
    _HERE / "germano_training_output_32_v3" / "checkpoints" / "best.pt"
)
CHECKPOINT = Path(os.environ.get("PBOUT_UNET_CHECKPOINT", _DEFAULT_CHECKPOINT))
_dns_path = os.environ.get("PBOUT_UNET_DNS_PATH")
DNS_PATH = Path(_dns_path).expanduser() if _dns_path else None
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
VORTICITY_MODE = _checkpoint.get("vorticity_mode", "potential_laplacian")
LEARN_VORTICITY = VORTICITY_MODE == "learned_channel"
OUTPUT_CHANNELS = int(
    _checkpoint.get("model_output_channels", 3 if LEARN_VORTICITY else 2)
)
if FINE_N % COARSE_N:
    raise ValueError("Checkpoint fine_n must be divisible by coarse_n")

MODEL = ResidualUNetSR(
    len(FIELD_NAMES), BASE_CHANNELS, FINE_N // COARSE_N, OUTPUT_CHANNELS
).to(DEVICE)
MODEL.load_state_dict(_checkpoint["model"])
MODEL.eval()
MODEL.requires_grad_(False)

MEAN = torch.as_tensor(_checkpoint["mean"], dtype=torch.float32, device=DEVICE)
STD = torch.as_tensor(_checkpoint["std"], dtype=torch.float32, device=DEVICE)
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


def _arakawa_bracket(f: torch.Tensor, g: torch.Tensor, dx: float, dy: float):
    """The three-Jacobian Arakawa stencil used by misc_styles.find_bracket."""
    tr = lambda a, i, j: torch.roll(a, shifts=(i, j), dims=(-2, -1))
    jpp = (tr(f, 0, 1) - tr(f, 0, -1)) * (tr(g, 1, 0) - tr(g, -1, 0)) \
        - (tr(f, 1, 0) - tr(f, -1, 0)) * (tr(g, 0, 1) - tr(g, 0, -1))
    jpx = (
        tr(g, 1, 0) * (tr(f, 1, 1) - tr(f, 1, -1))
        - tr(g, -1, 0) * (tr(f, -1, 1) - tr(f, -1, -1))
        - tr(g, 0, 1) * (tr(f, 1, 1) - tr(f, -1, 1))
        + tr(g, 0, -1) * (tr(f, 1, -1) - tr(f, -1, -1))
    )
    jxp = (
        tr(g, 1, 1) * (tr(f, 0, 1) - tr(f, 1, 0))
        - tr(g, -1, -1) * (tr(f, -1, 0) - tr(f, 0, -1))
        - tr(g, -1, 1) * (tr(f, 0, 1) - tr(f, -1, 0))
        + tr(g, 1, -1) * (tr(f, 1, 0) - tr(f, 0, -1))
    )
    return (jpp + jpx + jxp) / (12.0 * dx * dy)


def _germano_filter_torch(field: torch.Tensor) -> torch.Tensor:
    """Apply the checkpoint's forward differential/Germano filter."""
    nx, ny = field.shape[-2:]
    kx = 2 * torch.pi * torch.fft.fftfreq(nx, d=LX / nx, device=DEVICE)
    ky = 2 * torch.pi * torch.fft.fftfreq(ny, d=LY / ny, device=DEVICE)
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
def _reconstruct_dns(coarse: torch.Tensor, physical_length: float):
    """Return reconstructed DNS fields and the cutoff-filtered U-Net output."""
    # BOUT++ stores vorticity on its physical domain, whereas the checkpoint
    # was trained with the domain lengths LX and LY.  Convert the Laplacian
    # channel back to checkpoint units before normalising the network input.
    model_input = coarse.clone()
    model_input[:, 2:3] *= (physical_length / LX) ** 2
    normalised = (model_input - MEAN) / STD
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
    if VORTICITY_FROM_POTENTIAL:
        vorticity = periodic_laplacian_4th_torch(
            dns[:, 1:2], physical_length, physical_length
        )
        dns = torch.cat((dns[:, :2], vorticity), dim=1)
        return dns, inverse_input
    # Convert checkpoint-domain vorticity to the physical BOUT++ domain.
    vorticity_scale = (physical_length / LX) ** 2
    dns = torch.cat((dns[:, :2], dns[:, 2:3] / vorticity_scale), dim=1)
    return dns, inverse_input


def initFlow(npv):
    """Return initial LES fields, optionally derived from a DNS snapshot.

    If ``PBOUT_UNET_DNS_PATH`` is set, load the ``U`` (density) and ``V``
    (potential) arrays from that ``.npz`` DNS snapshot, apply the checkpoint's
    Germano filter, and spectrally resize them to the LES grid.  Otherwise,
    preserve the initial fields supplied by BOUT++.
    """
    values = np.asarray(npv, dtype=np.float64).reshape(-1)
    expected = 2 + 3 * COARSE_N * COARSE_N
    if values.size != expected:
        raise ValueError(
            f"initFlow received {values.size} values, expected {expected}. "
            "In hw.cxx set initFlow's NumPy dimension to 2 + 3*SIZE*SIZE."
        )
    if DNS_PATH is None:
        # initFlow's C++ payload is point-interleaved n, phi, vort.
        fields = values[2:].reshape(COARSE_N, COARSE_N, 3).transpose(2, 0, 1)
        initial_state = "BOUT++ fields"
    else:
        if not DNS_PATH.is_file():
            raise FileNotFoundError(
                f"DNS snapshot not found: {DNS_PATH}. "
                "Set PBOUT_UNET_DNS_PATH to an existing .npz file."
            )
        dns = load_dns_snapshot(
            DNS_PATH, FIELD_NAMES, FINE_N, LX, LY, dtype=np.float64
        )
        fields, _ = filter_snapshot(
            dns, ALPHA, LX, LY, COARSE_N, dtype=np.float64
        )
        initial_state = f"filtered DNS snapshot {DNS_PATH}"

    # Temporary output correction for the current BOUT++ coupling.
    # print(fields.shape)
    #fields = fields.copy()
    #fields[2] /= 64.0
    physical_length = float(values[0]) * COARSE_N
    fields[2:3, :, :] /= (physical_length / LX) ** 2

    # Save the DNS reconstruction of the exact LES state returned to BOUT++.
    coarse = torch.from_numpy(fields[None]).to(DEVICE, torch.float32)
    initial_dns, initial_fdns = _reconstruct_dns(coarse, physical_length)
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

    print(
        f"Loaded BOUT++ U-Net checkpoint {CHECKPOINT} on {DEVICE}; "
        f"LES={COARSE_N}, DNS={FINE_N}, implicit={IMPLICIT}; "
        f"vorticity_from_potential={VORTICITY_FROM_POTENTIAL}; "
        f"initial state={initial_state}; saved reconstruction={initial_output}; "
        f"saved U-Net output={initial_filtered_output}",
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
    required_channels = 5 if IMPLICIT else 3
    fields, batch = _decode_fields(values, 4, required_channels)

    coarse = torch.from_numpy(fields[:, :3]).to(DEVICE, torch.float32)
    dns, filtered_fine = _reconstruct_dns(coarse, dx_les * COARSE_N)
    if time_step % 1000 == 0:
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
        np.savez_compressed(
            results_dir / f"fields_fDNS_{(time_step+1):07d}.npz",
            pStep=np.asarray(time_step),
            simtime=np.asarray(simtime),
            U=filtered_output_fields[0],
            V=filtered_output_fields[1],
            P=filtered_output_fields[2],
        )

    dx_dns = dx_les * COARSE_N / FINE_N
    phi = dns[:, 1:2]
    bracket_vort = _arakawa_bracket(phi, dns[:, 2:3], dx_dns, dx_dns)
    bracket_n = _arakawa_bracket(phi, dns[:, 0:1], dx_dns, dx_dns)

    # misc_styles.find_bracket filters the fine-grid bracket before returning it
    # to LES.  Use the same Germano operator and the training spectral coarsening.
    bracket_vort = _spectral_resize_torch(
        _germano_filter_torch(bracket_vort), COARSE_N
    )[:, 0]
    bracket_n = _spectral_resize_torch(
        _germano_filter_torch(bracket_n), COARSE_N
    )[:, 0]

    if IMPLICIT:
        les_vort = torch.from_numpy(fields[:, 3]).to(DEVICE, torch.float32)
        les_n = torch.from_numpy(fields[:, 4]).to(DEVICE, torch.float32)
        bracket_vort = -(bracket_vort - les_vort)
        bracket_n = -(bracket_n - les_n)

    # Back to (x, batch/y, z), then C-order flattening expected by hw.cxx.
    out_vort = bracket_vort.permute(1, 0, 2).reshape(-1)
    out_n = bracket_n.permute(1, 0, 2).reshape(-1)
    iteration = np.asarray([0.0], dtype=np.float64)
    return np.concatenate(
        (iteration, out_vort.cpu().double().numpy(), out_n.cpu().double().numpy())
    )
