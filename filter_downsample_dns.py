#!/usr/bin/env python3
"""Apply the differential filter and spectral downsampling to an HW snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from germano_common import (
    HW_FIELD_LABELS,
    HW_FIELD_NAMES,
    filter_snapshot,
    hw_fields_numpy,
    isotropic_spectrum,
    k2_numpy,
    load_dns_snapshot,
    save_image,
    spectral_resize,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("filter_downsample_output"),
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help=(
            "Filter width in DNS grid spacings; use 0 to disable the "
            "differential filter (default: 1.0)"
        ),
    )
    parser.add_argument("--coarse-n", type=int, default=32)
    parser.add_argument("--lx", type=float, default=2 * np.pi)
    parser.add_argument("--ly", type=float, default=2 * np.pi)
    return parser.parse_args()


def infer_dns_resolution(path):
    with np.load(path) as source:
        if "U" not in source:
            raise KeyError(f"{path} is missing the U field.")
        shape = source["U"].shape
    if len(shape) != 2 or shape[0] != shape[1]:
        raise ValueError(
            f"Expected a square two-dimensional field, got {shape}."
        )
    return shape[0]


def gaussian_filter(field, alpha, lx, ly):
    """Apply a periodic Gaussian filter with standard deviation alpha."""
    transfer = np.exp(-0.5 * alpha**2 * k2_numpy(field.shape[0], lx, ly))
    return np.fft.ifft2(np.fft.fft2(field) * transfer).real


def gaussian_filter_snapshot(dns, alpha, lx, ly, coarse_n):
    filtered_density = gaussian_filter(dns[0], alpha, lx, ly)
    filtered_potential = gaussian_filter(dns[1], alpha, lx, ly)
    filtered = hw_fields_numpy(
        filtered_density,
        filtered_potential,
        lx,
        ly,
    )
    coarse_density = spectral_resize(filtered[0], coarse_n)
    coarse_potential = spectral_resize(filtered[1], coarse_n)
    downsampled = hw_fields_numpy(
        coarse_density,
        coarse_potential,
        lx,
        ly,
    )
    return downsampled, filtered


def save_spectra(
    name,
    dns,
    differential_filtered,
    differential_downsampled,
    gaussian_filtered,
    gaussian_downsampled,
    coarse_n,
    lx,
    ly,
    path,
):
    spectra = {
        "DNS": isotropic_spectrum(dns, lx, ly),
        f"Differential filtered {dns.shape[0]}": isotropic_spectrum(
            differential_filtered, lx, ly
        ),
        f"Differential filtered {coarse_n}": isotropic_spectrum(
            differential_downsampled, lx, ly
        ),
        f"Gaussian filtered {dns.shape[0]}": isotropic_spectrum(
            gaussian_filtered, lx, ly
        ),
        f"Gaussian filtered {coarse_n}": isotropic_spectrum(
            gaussian_downsampled, lx, ly
        ),
    }
    figure, axis = plt.subplots(figsize=(7.4, 5.6))
    for label, (wavenumber, power) in spectra.items():
        valid = (wavenumber > 0) & (power > 0)
        if label.endswith(f"filtered {coarse_n}"):
            axis.loglog(
                wavenumber[valid],
                power[valid],
                linestyle="None",
                marker="o",
                markersize=4,
                label=label,
            )
        else:
            axis.loglog(
                wavenumber[valid],
                power[valid],
                linewidth=1.8,
                label=label,
            )
    nyquist = min(np.pi / (lx / coarse_n), np.pi / (ly / coarse_n))
    axis.axvline(
        nyquist,
        linestyle=":",
        linewidth=1.4,
        label=f"{coarse_n}-grid Nyquist",
    )
    axis.set_title(f"{HW_FIELD_LABELS[name]}: isotropic spectrum comparison")
    axis.set_xlabel(r"$k$")
    axis.set_ylabel(r"Shell power $P(k)$")
    axis.set_ylim(bottom=10e-10)
    axis.grid(True, which="both", alpha=0.3)
    axis.legend(numpoints=1)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    args = parse_args()
    if args.alpha < 0:
        raise ValueError("alpha must be non-negative.")
    if args.coarse_n < 5:
        raise ValueError("coarse-n must be at least 5 for the W stencil.")
    fine_n = infer_dns_resolution(args.input)
    if args.coarse_n >= fine_n:
        raise ValueError("coarse-n must be smaller than the DNS resolution.")
    delta = args.lx / fine_n
    effective_alpha = args.alpha * delta

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dns = load_dns_snapshot(
        args.input,
        HW_FIELD_NAMES,
        fine_n,
        args.lx,
        args.ly,
    )
    differential_downsampled, differential_filtered = filter_snapshot(
        dns,
        effective_alpha,
        args.lx,
        args.ly,
        args.coarse_n,
    )
    gaussian_downsampled, gaussian_filtered = gaussian_filter_snapshot(
        dns,
        effective_alpha,
        args.lx,
        args.ly,
        args.coarse_n,
    )

    colormaps = {"U": "Blues", "V": "Reds_r", "W": "hot"}
    for channel, name in enumerate(HW_FIELD_NAMES):
        label = HW_FIELD_LABELS[name]
        save_image(
            dns[channel],
            f"{label}: DNS (original DNS)",
            args.output_dir / f"{name}_DNS.png",
            colormaps[name],
        )
        save_image(
            differential_filtered[channel],
            f"{label}: Differential filtered {fine_n}",
            args.output_dir / f"{name}_Differential_filtered_{fine_n}.png",
            colormaps[name],
        )
        save_image(
            differential_downsampled[channel],
            f"{label}: Differential filtered {args.coarse_n}",
            args.output_dir
            / f"{name}_Differential_filtered_{args.coarse_n}.png",
            colormaps[name],
        )
        save_image(
            gaussian_filtered[channel],
            f"{label}: Gaussian filtered {fine_n}",
            args.output_dir / f"{name}_Gaussian_filtered_{fine_n}.png",
            colormaps[name],
        )
        save_image(
            gaussian_downsampled[channel],
            f"{label}: Gaussian filtered {args.coarse_n}",
            args.output_dir / f"{name}_Gaussian_filtered_{args.coarse_n}.png",
            colormaps[name],
        )
        save_spectra(
            name,
            dns[channel],
            differential_filtered[channel],
            differential_downsampled[channel],
            gaussian_filtered[channel],
            gaussian_downsampled[channel],
            args.coarse_n,
            args.lx,
            args.ly,
            args.output_dir / f"{name}_spectrum_comparison.png",
        )

    np.savez_compressed(
        args.output_dir / "filtered_fields.npz",
        dns=dns,
        differential_filtered=differential_filtered,
        differential_downsampled=differential_downsampled,
        gaussian_filtered=gaussian_filtered,
        gaussian_downsampled=gaussian_downsampled,
        field_names=np.asarray(HW_FIELD_NAMES),
    )
    metadata = {
        "input": str(args.input),
        "alpha_factor": args.alpha,
        "delta": delta,
        "effective_alpha": effective_alpha,
        "differential_transfer": "1 / (1 + alpha^2 k^2)",
        "gaussian_transfer": "exp(-0.5 alpha^2 k^2)",
        "fine_n": fine_n,
        "coarse_n": args.coarse_n,
        "lx": args.lx,
        "ly": args.ly,
        "downsampling": "centered Fourier-mode truncation",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )
    print(f"Saved filtered fields and plots to {args.output_dir}")


if __name__ == "__main__":
    main()
