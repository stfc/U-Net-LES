#!/usr/bin/env python3
"""
Germano filter -> 256-to-32 spectral restriction -> 32-to-256 spectral
interpolation -> inverse Germano filter, for periodic 2-D fields in an NPZ file.

Default:
    alpha = alpha_factor * delta
    delta = Lx / Nfine
    alpha_factor = 1.0

Changing alpha later
--------------------
Use either:

    python germano_npz_pipeline.py input.npz --alpha-factor 2.0

to set alpha = 2*delta, or:

    python germano_npz_pipeline.py input.npz --alpha 0.05

to provide alpha directly. An explicit --alpha overrides --alpha-factor.

Example
-------
    python germano_npz_pipeline.py fields_run4_1545te.npz \
        --coarse-n 32 \
        --alpha-factor 1.0 \
        --lx 6.283185307179586 \
        --ly 6.283185307179586
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def wavenumber_squared(
    nx: int,
    ny: int,
    lx: float,
    ly: float,
) -> np.ndarray:
    """Return kx^2 + ky^2 using angular wavenumbers."""
    dx = lx / nx
    dy = ly / ny

    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)

    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    return kx_grid**2 + ky_grid**2


def germano_filter(
    field: np.ndarray,
    alpha: float,
    lx: float,
    ly: float,
) -> np.ndarray:
    """
    Apply the Germano differential filter:

        (I - alpha^2 Laplacian) u_bar = u

    Fourier representation:

        u_bar_hat = u_hat / (1 + alpha^2 |k|^2)
    """
    nx, ny = field.shape
    k2 = wavenumber_squared(nx, ny, lx, ly)
    field_hat = np.fft.fft2(field)
    filtered_hat = field_hat / (1.0 + alpha**2 * k2)
    return np.fft.ifft2(filtered_hat).real


def inverse_germano_filter(
    filtered_field: np.ndarray,
    alpha: float,
    lx: float,
    ly: float,
) -> np.ndarray:
    """
    Apply the exact inverse differential operator:

        u = (I - alpha^2 Laplacian) u_bar

    Fourier representation:

        u_hat = (1 + alpha^2 |k|^2) u_bar_hat
    """
    nx, ny = filtered_field.shape
    k2 = wavenumber_squared(nx, ny, lx, ly)
    filtered_hat = np.fft.fft2(filtered_field)
    recovered_hat = (1.0 + alpha**2 * k2) * filtered_hat
    return np.fft.ifft2(recovered_hat).real


def spectral_resize(field: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """
    Resize a periodic 2-D field by centred Fourier truncation or zero-padding.

    This function is intended for even source and target dimensions.
    The FFT scaling correction preserves Fourier-series amplitudes.
    """
    source_nx, source_ny = field.shape
    target_nx, target_ny = target_shape

    if min(source_nx, source_ny, target_nx, target_ny) <= 0:
        raise ValueError("All grid dimensions must be positive.")

    source_hat = np.fft.fftshift(np.fft.fft2(field))
    target_hat = np.zeros((target_nx, target_ny), dtype=np.complex128)

    copy_nx = min(source_nx, target_nx)
    copy_ny = min(source_ny, target_ny)

    src_x0 = (source_nx - copy_nx) // 2
    src_y0 = (source_ny - copy_ny) // 2
    dst_x0 = (target_nx - copy_nx) // 2
    dst_y0 = (target_ny - copy_ny) // 2

    target_hat[
        dst_x0 : dst_x0 + copy_nx,
        dst_y0 : dst_y0 + copy_ny,
    ] = source_hat[
        src_x0 : src_x0 + copy_nx,
        src_y0 : src_y0 + copy_ny,
    ]

    target_hat *= (target_nx * target_ny) / (source_nx * source_ny)

    resized = np.fft.ifft2(np.fft.ifftshift(target_hat))
    return resized.real


def relative_l2(reference: np.ndarray, approximation: np.ndarray) -> float:
    denominator = np.linalg.norm(reference.ravel())
    if denominator == 0.0:
        return float(np.linalg.norm(approximation.ravel()))
    return float(
        np.linalg.norm((approximation - reference).ravel()) / denominator
    )


def rmse(reference: np.ndarray, approximation: np.ndarray) -> float:
    return float(np.sqrt(np.mean((approximation - reference) ** 2)))


def correlation(reference: np.ndarray, approximation: np.ndarray) -> float:
    ref = reference.ravel()
    app = approximation.ravel()
    if np.std(ref) == 0.0 or np.std(app) == 0.0:
        return float("nan")
    return float(np.corrcoef(ref, app)[0, 1])



def isotropic_energy_spectrum(
    field: np.ndarray,
    lx: float,
    ly: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return shell-integrated isotropic Fourier power P(k)."""
    nx, ny = field.shape
    dx = lx / nx
    dy = ly / ny
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=dy)
    kx_grid, ky_grid = np.meshgrid(kx, ky, indexing="ij")
    kmag = np.sqrt(kx_grid**2 + ky_grid**2)
    field_hat = np.fft.fft2(field)
    modal_power = np.abs(field_hat) ** 2 / (nx * ny) ** 2
    dk = min(2.0 * np.pi / lx, 2.0 * np.pi / ly)
    shell_index = np.rint(kmag / dk).astype(np.int64)
    shell_power = np.bincount(shell_index.ravel(), weights=modal_power.ravel())
    shell_count = np.bincount(shell_index.ravel())
    k_shell = np.arange(shell_power.size, dtype=np.float64) * dk
    valid = shell_count > 0
    return k_shell[valid], shell_power[valid]


def save_spectrum_comparison(
    name: str,
    fields: dict[str, np.ndarray],
    output_path: Path,
    csv_path: Path,
    lx: float,
    ly: float,
    coarse_n: int,
) -> None:
    """Generate and save the spectrum-comparison plot and CSV."""
    spectra = {label: isotropic_energy_spectrum(values, lx, ly) for label, values in fields.items()}
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for label, (k, power) in spectra.items():
        mask = (k > 0.0) & (power > 0.0)

        if label == "Filtered 32":
            ax.loglog(
                k[mask],
                power[mask],
                linestyle="None",
                marker="o",
                markersize=2.5,
                label=label,
            )
        elif label == "Filtered 32":
            ax.loglog(
                k[mask],
                power[mask],
                linestyle="None",
                marker="o",
                markersize=2.5,
                label=label,
            )
        else:
            ax.loglog(
                k[mask],
                power[mask],
                linewidth=1.8,
                label=label,
            )
    coarse_nyquist = min(np.pi / (lx / coarse_n), np.pi / (ly / coarse_n))
    ax.axvline(coarse_nyquist, linestyle="--", linewidth=1.2, label=f"{coarse_n}-grid Nyquist")
    ax.set_title(f"{name}: isotropic spectrum comparison")
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"Shell power $P(k)$")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    ax.set_ylim(bottom=1e-18)
    #ax.loglog(k[mask], power[mask], '.', markersize=5, label=label,)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    all_k = sorted(set().union(*(set(k.tolist()) for k, _ in spectra.values())))
    lookups = {label: {float(kv): float(pv) for kv, pv in zip(k, power)} for label, (k, power) in spectra.items()}
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["k"] + list(spectra.keys()))
        writer.writeheader()
        for kval in all_k:
            row = {"k": kval}
            for label in spectra:
                row[label] = lookups[label].get(kval, "")
            writer.writerow(row)


def save_single_field_plot(
    field: np.ndarray,
    title: str,
    output_path: Path,
    lx: float,
    ly: float,
) -> None:
    """Save one field per figure."""
    fig, ax = plt.subplots(figsize=(6.4, 5.3))
    image = ax.imshow(
        field.T,
        origin="lower",
        extent=(0.0, lx, 0.0, ly),
        aspect="equal",
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def process_field(
    name: str,
    field: np.ndarray,
    coarse_n: int,
    alpha: float,
    lx: float,
    ly: float,
    output_dir: Path,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    if field.ndim != 2:
        raise ValueError(f"Field {name!r} must be 2-D, got shape {field.shape}")

    nx, ny = field.shape
    if nx != ny:
        raise ValueError(
            f"Current pipeline expects square fields; {name!r} has {field.shape}"
        )

    filtered_fine = germano_filter(field, alpha, lx, ly)
    filtered_coarse = spectral_resize(filtered_fine, (coarse_n, coarse_n))
    filtered_interpolated = spectral_resize(filtered_coarse, (nx, ny))

    print(np.max(np.abs(filtered_fine - filtered_interpolated)))

    #filtered_interpolated = filtered_fine

    reconstructed = inverse_germano_filter(
        filtered_interpolated, alpha, lx, ly
    )

    filtered_error = filtered_interpolated - filtered_fine
    reconstruction_error = reconstructed - field

    arrays = {
        f"{name}_dns": field,
        f"{name}_filtered_{nx}": filtered_fine,
        f"{name}_filtered_{coarse_n}": filtered_coarse,
        f"{name}_spectral_interp_{nx}": filtered_interpolated,
        f"{name}_reconstructed_{nx}": reconstructed,
        f"{name}_filtered_error": filtered_error,
        f"{name}_reconstruction_error": reconstruction_error,
    }

    metrics = {
        "field": name,
        "filtered_relative_l2": relative_l2(
            filtered_fine, filtered_interpolated
        ),
        "filtered_rmse": rmse(filtered_fine, filtered_interpolated),
        "filtered_max_abs_error": float(np.max(np.abs(filtered_error))),
        "dns_relative_l2": relative_l2(field, reconstructed),
        "dns_rmse": rmse(field, reconstructed),
        "dns_max_abs_error": float(np.max(np.abs(reconstruction_error))),
        "dns_correlation": correlation(field, reconstructed),
    }

    plots = {
        "dns": field,
        f"filtered_{nx}": filtered_fine,
        f"filtered_{coarse_n}": filtered_coarse,
        f"spectral_interp_{nx}": filtered_interpolated,
        f"reconstructed_LES_{nx}": filtered_interpolated,
        f"reconstructed_{nx}": reconstructed,
        "reconstruction_error": reconstruction_error,
    }

    for stage, values in plots.items():
        save_single_field_plot(
            values,
            title=f"{name}: {stage.replace('_', ' ')}",
            output_path=output_dir / f"{name}_{stage}.png",
            lx=lx,
            ly=ly,
        )

    save_spectrum_comparison(
        name=name,
        fields={
            "DNS": field,
            "Filtered 256": filtered_fine,
            "Filtered 32": filtered_coarse,  # add this
            "Spectral interpolation": filtered_interpolated,
            "Reconstructed DNS": reconstructed,
        },
        output_path=output_dir / f"{name}_spectrum_comparison.png",
        csv_path=output_dir / f"{name}_spectra.csv",
        lx=lx,
        ly=ly,
        coarse_n=coarse_n,
    )

    return arrays, metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the periodic Germano filter/reconstruction pipeline."
    )
    parser.add_argument("input", type=Path, help="Input NPZ file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("germano_npz_output"),
    )
    parser.add_argument("--coarse-n", type=int, default=32)
    parser.add_argument("--lx", type=float, default=2.0 * np.pi)
    parser.add_argument("--ly", type=float, default=2.0 * np.pi)

    alpha_group = parser.add_mutually_exclusive_group()
    alpha_group.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Absolute Germano filter width. Overrides --alpha-factor.",
    )
    alpha_group.add_argument(
        "--alpha-factor",
        type=float,
        default=1.0,
        help="Set alpha = alpha_factor * delta, with delta=Lx/Nfine.",
    )

    parser.add_argument(
        "--fields",
        nargs="+",
        default=None,
        help="NPZ keys to process. Default: all 2-D square arrays.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(args.input)
    if args.coarse_n <= 0:
        raise ValueError("--coarse-n must be positive.")
    if args.lx <= 0.0 or args.ly <= 0.0:
        raise ValueError("Domain lengths must be positive.")

    source = np.load(args.input)
    selected = args.fields if args.fields is not None else source.files

    valid_fields: dict[str, np.ndarray] = {}
    for name in selected:
        if name not in source:
            raise KeyError(f"Field {name!r} is not present in {args.input}")
        array = np.asarray(source[name])
        if array.ndim == 2 and array.shape[0] == array.shape[1]:
            valid_fields[name] = array.astype(np.float64, copy=False)

    if not valid_fields:
        raise ValueError("No square 2-D fields were found.")

    fine_sizes = {array.shape[0] for array in valid_fields.values()}
    if len(fine_sizes) != 1:
        raise ValueError("All selected fields must have the same resolution.")

    fine_n = fine_sizes.pop()
    if args.coarse_n >= fine_n:
        raise ValueError("--coarse-n must be smaller than the fine resolution.")

    delta_x = args.lx / fine_n
    delta_y = args.ly / fine_n

    if args.alpha is not None:
        alpha = args.alpha
        alpha_definition = "absolute"
        alpha_factor = alpha / delta_x
    else:
        alpha_factor = args.alpha_factor
        alpha = alpha_factor * delta_x
        alpha_definition = "alpha_factor * delta_x"

    if alpha < 0.0:
        raise ValueError("alpha must be non-negative.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    output_arrays: dict[str, np.ndarray] = {}
    all_metrics: list[dict[str, float]] = []

    for name, field in valid_fields.items():
        arrays, metrics = process_field(
            name=name,
            field=field,
            coarse_n=args.coarse_n,
            alpha=alpha,
            lx=args.lx,
            ly=args.ly,
            output_dir=args.output_dir,
        )
        output_arrays.update(arrays)
        all_metrics.append(metrics)

    metadata = {
        "input_file": str(args.input),
        "fields": list(valid_fields),
        "fine_n": fine_n,
        "coarse_n": args.coarse_n,
        "lx": args.lx,
        "ly": args.ly,
        "delta_x": delta_x,
        "delta_y": delta_y,
        "alpha": alpha,
        "alpha_factor_relative_to_delta_x": alpha_factor,
        "alpha_definition": alpha_definition,
    }

    output_arrays["metadata_json"] = np.array(json.dumps(metadata))
    npz_path = args.output_dir / "germano_pipeline_results.npz"
    np.savez_compressed(npz_path, **output_arrays)

    csv_path = args.output_dir / "metrics.csv"
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(all_metrics[0].keys()))
        writer.writeheader()
        writer.writerows(all_metrics)

    json_path = args.output_dir / "metadata.json"
    json_path.write_text(json.dumps(metadata, indent=2))

    print("\nConfiguration")
    print(json.dumps(metadata, indent=2))
    print("\nMetrics")
    for item in all_metrics:
        print(
            f"{item['field']}: "
            f"filtered relL2={item['filtered_relative_l2']:.6e}, "
            f"DNS relL2={item['dns_relative_l2']:.6e}, "
            f"DNS correlation={item['dns_correlation']:.6f}"
        )

    print(f"\nResults: {npz_path}")
    print(f"Metrics: {csv_path}")
    print(f"Plots:   {args.output_dir}")


if __name__ == "__main__":
    main()