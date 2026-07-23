#!/usr/bin/env python3
"""Shared reconstructed-field and Arakawa visualization routines."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from germano_common import HW_FIELD_LABELS, isotropic_spectrum

SPECTRUM_Y_MIN = 10e-10


def save_field_images(
    fields: np.ndarray,
    field_names,
    output_path: Path,
    title_prefix: str = "Reconstructed DNS",
) -> Path:
    """Save channel-first physical fields in a single diagnostic image."""
    if fields.ndim != 3 or fields.shape[0] != len(field_names):
        raise ValueError(
            "fields must have shape (len(field_names), nx, ny)"
        )
    nx, ny = fields.shape[-2:]
    fig, axes = plt.subplots(
        1, len(field_names),
        figsize=(5.8 * len(field_names), 5.2),
        squeeze=False,
    )
    for channel, (ax, name) in enumerate(zip(axes[0], field_names)):
        cmap = {
            "U": "Blues",
            "V": "Reds_r",
            "W": "hot",
            "P": "hot",
        }.get(name, "viridis")
        image = ax.imshow(
            fields[channel].T,
            origin="lower",
            extent=(0, nx, 0, ny),
            aspect="equal",
            cmap=cmap,
        )
        ax.set_title(f"{title_prefix}: {HW_FIELD_LABELS.get(name, name)}")
        ax.set_xlabel("x [pixels]")
        ax.set_ylabel("y [pixels]")
        fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def arakawa_bracket(
    f: torch.Tensor, g: torch.Tensor, dx: float, dy: float
) -> torch.Tensor:
    """Return the periodic three-Jacobian Arakawa bracket."""
    tr = lambda a, i, j: torch.roll(a, shifts=(i, j), dims=(-2, -1))
    jpp = (
        (tr(f, 0, 1) - tr(f, 0, -1)) * (tr(g, 1, 0) - tr(g, -1, 0))
        - (tr(f, 1, 0) - tr(f, -1, 0)) * (tr(g, 0, 1) - tr(g, 0, -1))
    )
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


def save_field_spectrum_comparison(
    reconstructed: np.ndarray,
    dns: np.ndarray,
    field_names,
    lx: float,
    ly: float,
    output_path: Path,
    cutoff_fraction: float | None = None,
) -> Path:
    """Plot every reconstructed field spectrum against DNS."""
    count = len(field_names)
    fig, axes = plt.subplots(1, count, figsize=(5.8 * count, 5.2), squeeze=False)
    for channel, (ax, name) in enumerate(zip(axes[0], field_names)):
        for values, label, color in (
            (dns[channel], "DNS", "black"),
            (reconstructed[channel], "Reconstructed DNS", "tab:blue"),
        ):
            k, power = isotropic_spectrum(values, lx, ly)
            positive = (k > 0) & (power > 0)
            ax.loglog(k[positive], power[positive], color=color, label=label)
        _add_cutoff(ax, reconstructed.shape[-2:], lx, ly, cutoff_fraction)
        ax.set_title(f"{HW_FIELD_LABELS.get(name, name)} spectrum")
        ax.set_xlabel(r"wavenumber $k$")
        ax.set_ylabel("shell-summed power")
        ax.set_ylim(bottom=SPECTRUM_Y_MIN)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def calculate_arakawa_terms(
    fields: np.ndarray, lx: float, ly: float
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate J(phi, omega) and J(phi, density) for U,V,W fields."""
    tensor = torch.as_tensor(fields[None])
    potential = tensor[:, 1:2]
    dx, dy = lx / fields.shape[-2], ly / fields.shape[-1]
    terms = (
        arakawa_bracket(potential, tensor[:, 2:3], dx, dy),
        arakawa_bracket(potential, tensor[:, 0:1], dx, dy),
    )
    return tuple(term[0, 0].detach().cpu().double().numpy() for term in terms)


def _add_cutoff(ax, shape, lx, ly, cutoff_fraction):
    if cutoff_fraction is None:
        return
    nyquist = min(np.pi / (lx / shape[0]), np.pi / (ly / shape[1]))
    ax.axvline(
        cutoff_fraction * nyquist, color="tab:red", linestyle="--",
        label=f"cutoff ({cutoff_fraction:g} Nyquist)",
    )


def save_arakawa_comparison(
    reconstructed: np.ndarray,
    dns: np.ndarray,
    lx: float,
    ly: float,
    output_dir: Path,
    stem: str = "arakawa_DNS",
    cutoff_fraction: float | None = None,
) -> tuple[Path, Path]:
    """Save reconstructed Arakawa fields and their spectra compared with DNS."""
    output_dir.mkdir(parents=True, exist_ok=True)
    current = calculate_arakawa_terms(reconstructed, lx, ly)
    reference = calculate_arakawa_terms(dns, lx, ly)
    titles = (r"$J(V,W)$", r"$J(V,U)$")
    nx, ny = reconstructed.shape[-2:]
    center = ny // 2

    field_path = output_dir / f"{stem}.png"
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.2))
    for index, (values, dns_values, title) in enumerate(
        zip(current, reference, titles)
    ):
        maximum = float(np.max(np.abs(values)))
        limits = {"vmin": -maximum, "vmax": maximum} if maximum > 0 else {}
        image = axes[0, index].imshow(
            values.T, origin="lower", extent=(0, nx, 0, ny),
            aspect="equal", cmap="coolwarm", **limits,
        )
        axes[0, index].set_title(f"Reconstructed DNS {title}")
        axes[0, index].set_xlabel("x [pixels]")
        axes[0, index].set_ylabel("y [pixels]")
        fig.colorbar(image, ax=axes[0, index])
        axes[1, index].plot(dns_values[:, center], color="black", label="DNS")
        axes[1, index].plot(
            values[:, center], color="tab:blue", label="Reconstructed DNS"
        )
        axes[1, index].set_title(f"{title}: centerline y={center}")
        axes[1, index].set_xlabel("x [pixels]")
        axes[1, index].set_ylabel("Arakawa term")
        axes[1, index].grid(True, alpha=0.25)
        axes[1, index].legend()
    fig.tight_layout()
    fig.savefig(field_path, dpi=180)
    plt.close(fig)

    spectrum_path = output_dir / f"{stem}_spectrum_comparison.png"
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2))
    for ax, values, dns_values, title in zip(axes, current, reference, titles):
        for field, label, color in (
            (dns_values, "DNS", "black"),
            (values, "Reconstructed DNS", "tab:blue"),
        ):
            k, power = isotropic_spectrum(field, lx, ly)
            positive = (k > 0) & (power > 0)
            ax.loglog(k[positive], power[positive], color=color, label=label)
        _add_cutoff(ax, (nx, ny), lx, ly, cutoff_fraction)
        ax.set_title(f"{title} spectrum")
        ax.set_xlabel(r"wavenumber $k$")
        ax.set_ylabel("shell-summed power")
        ax.set_ylim(bottom=SPECTRUM_Y_MIN)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(spectrum_path, dpi=180)
    plt.close(fig)
    return field_path, spectrum_path
