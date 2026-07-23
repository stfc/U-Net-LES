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
    """Save channel-first physical fields and their horizontal centerlines."""
    if fields.ndim != 3 or fields.shape[0] != len(field_names):
        raise ValueError(
            "fields must have shape (len(field_names), nx, ny)"
        )
    nx, ny = fields.shape[-2:]
    center = ny // 2
    x_pixels = np.arange(nx)
    fig, axes = plt.subplots(
        2, len(field_names),
        figsize=(5.8 * len(field_names), 9.2),
        squeeze=False,
    )
    for channel, name in enumerate(field_names):
        ax = axes[0, channel]
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
        center_ax = axes[1, channel]
        center_ax.plot(
            x_pixels, fields[channel, :, center], linewidth=1.2
        )
        center_ax.axhline(0.0, color="0.5", linewidth=0.7)
        center_ax.set_title(
            f"{HW_FIELD_LABELS.get(name, name)}: centerline y={center}"
        )
        center_ax.set_xlabel("x [pixels]")
        center_ax.set_ylabel(HW_FIELD_LABELS.get(name, name))
        center_ax.grid(True, alpha=0.25)
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


def _plot_arakawa_fields(
    fields: tuple[np.ndarray, np.ndarray],
    titles: tuple[str, str],
    figure_title: str,
    output_path: Path,
    reference_fields: tuple[np.ndarray, np.ndarray] | None = None,
    current_label: str | None = None,
    reference_label: str = "DNS",
) -> Path:
    """Save two Arakawa fields with horizontal centerline diagnostics."""
    nx, ny = fields[0].shape
    center = ny // 2
    x_pixels = np.arange(nx)
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.2))
    for column, (values, title) in enumerate(zip(fields, titles)):
        maximum = float(np.max(np.abs(values)))
        limits = {"vmin": -maximum, "vmax": maximum} if maximum > 0 else {}
        image = axes[0, column].imshow(
            values.T,
            origin="lower",
            extent=(0, nx, 0, ny),
            aspect="equal",
            cmap="coolwarm",
            **limits,
        )
        axes[0, column].set_title(f"{figure_title} {title}")
        axes[0, column].set_xlabel("x [pixels]")
        axes[0, column].set_ylabel("y [pixels]")
        fig.colorbar(image, ax=axes[0, column])

        center_ax = axes[1, column]
        if reference_fields is not None:
            center_ax.plot(
                x_pixels,
                reference_fields[column][:, center],
                color="black",
                linewidth=1.0,
                label=reference_label,
            )
        center_ax.plot(
            x_pixels,
            values[:, center],
            color="tab:blue",
            linewidth=1.2,
            label=current_label,
        )
        center_ax.axhline(0.0, color="0.5", linewidth=0.7)
        center_ax.set_title(f"{title}: centerline y={center}")
        center_ax.set_xlabel("x [pixels]")
        center_ax.set_ylabel("Arakawa term")
        center_ax.grid(True, alpha=0.25)
        if current_label is not None or reference_fields is not None:
            center_ax.legend()
    fig.suptitle(figure_title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path


def save_arakawa_diagnostics(
    time_step: int,
    diagnostic_sets,
    output_dir: Path,
) -> None:
    """Save batched LES, DNS, and filtered-DNS Arakawa diagnostics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    titles = (r"$J(\phi,\omega)$", r"$J(\phi,n)$")
    for name, figure_title, vorticity, density in diagnostic_sets:
        batch_count = len(vorticity)
        for batch_index, fields in enumerate(zip(vorticity, density)):
            suffix = "" if batch_count == 1 else f"_batch{batch_index:03d}"
            _plot_arakawa_fields(
                fields,
                titles,
                f"{figure_title}, step {time_step}",
                output_dir
                / f"arakawa_{name}_terms_time{time_step:07d}{suffix}.png",
            )


def save_arakawa_implementation_comparison(
    time_step: int,
    calculated_terms: tuple[np.ndarray, np.ndarray],
    bout_terms: tuple[np.ndarray, np.ndarray],
    output_dir: Path,
) -> None:
    """Compare Python and BOUT++ Arakawa fields and centerlines."""
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = (
        ("vorticity", calculated_terms[0], bout_terms[0], r"$J(V,P)$"),
        ("density", calculated_terms[1], bout_terms[1], r"$J(V,U)$"),
    )
    for name, calculated, bout, title in diagnostics:
        for batch_index, (calculated_field, bout_field) in enumerate(
            zip(calculated, bout)
        ):
            difference = calculated_field - bout_field
            fields = (calculated_field, bout_field, difference)
            labels = (
                "Calculated in pBOUT-unet",
                "Provided by BOUT++",
                "Calculated - BOUT++",
            )
            nx, ny = calculated_field.shape
            center = ny // 2
            x_pixels = np.arange(nx)
            fig, axes = plt.subplots(2, 3, figsize=(17.5, 9.2))
            for column, (values, label) in enumerate(zip(fields, labels)):
                maximum = float(np.max(np.abs(values)))
                limits = (
                    {"vmin": -maximum, "vmax": maximum}
                    if maximum > 0 else {}
                )
                image = axes[0, column].imshow(
                    values.T,
                    origin="lower",
                    extent=(0, nx, 0, ny),
                    aspect="equal",
                    cmap="coolwarm",
                    **limits,
                )
                axes[0, column].set_title(f"{title}: {label}")
                axes[0, column].set_xlabel("x [pixels]")
                axes[0, column].set_ylabel("y [pixels]")
                fig.colorbar(image, ax=axes[0, column])
                axes[1, column].plot(
                    x_pixels, values[:, center], linewidth=1.2
                )
                axes[1, column].axhline(
                    0.0, color="0.5", linewidth=0.7
                )
                axes[1, column].set_title(
                    f"{label}: centerline y={center}"
                )
                axes[1, column].set_xlabel("x [pixels]")
                axes[1, column].set_ylabel("Arakawa term")
                axes[1, column].grid(True, alpha=0.25)
            fig.suptitle(
                f"LES Arakawa implementation comparison, step {time_step}"
            )
            fig.tight_layout()
            suffix = (
                "" if len(calculated) == 1 else f"_batch{batch_index:03d}"
            )
            fig.savefig(
                output_dir
                / f"arakawa_LES_{name}_comparison_time{time_step:07d}"
                  f"{suffix}.png",
                dpi=180,
            )
            plt.close(fig)


def save_bout_arakawa_image(
    terms: np.ndarray,
    time_step: int,
    simtime: float,
    output_path: Path,
) -> Path:
    """Save BOUT++ Arakawa terms and their horizontal centerlines."""
    return _plot_arakawa_fields(
        (terms[0], terms[1]),
        (r"$J(\phi,\omega)$", r"$J(\phi,n)$"),
        f"DNS Arakawa terms, step {time_step}, t={simtime:g}",
        output_path,
    )


def save_arakawa_comparison(
    reconstructed: np.ndarray,
    dns: np.ndarray | None,
    lx: float,
    ly: float,
    output_dir: Path,
    stem: str = "arakawa_DNS",
    cutoff_fraction: float | None = None,
    spectrum_stem: str | None = None,
) -> tuple[Path, Path]:
    """Save Arakawa fields and spectra, optionally compared with DNS."""
    output_dir.mkdir(parents=True, exist_ok=True)
    current = calculate_arakawa_terms(reconstructed, lx, ly)
    reference = (
        calculate_arakawa_terms(dns, lx, ly) if dns is not None else None
    )
    titles = (r"$J(V,W)$", r"$J(V,U)$")
    nx, ny = reconstructed.shape[-2:]

    field_path = output_dir / f"{stem}.png"
    current_label = "Reconstructed DNS" if reference is not None else "DNS"
    _plot_arakawa_fields(
        current,
        titles,
        current_label,
        field_path,
        reference_fields=reference,
        current_label=current_label,
        reference_label="Original DNS",
    )

    spectrum_path = output_dir / (
        spectrum_stem or f"{stem}_spectrum_comparison.png"
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.2))
    for index, (ax, values, title) in enumerate(zip(axes, current, titles)):
        plotted_fields = [(values, current_label, "tab:blue")]
        if reference is not None:
            plotted_fields.insert(
                0, (reference[index], "Original DNS", "black")
            )
        for field, label, color in plotted_fields:
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
