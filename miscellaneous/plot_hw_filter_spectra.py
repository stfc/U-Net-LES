#!/usr/bin/env python3
"""Compare Gaussian and Germano filtering of a periodic HW snapshot.

Uses U=density and V=potential; E(k) is shell-summed HW energy.
Field multipliers are exp(-alpha**2*k**2) and 1/(1+alpha**2*k**2),
matching at small k (Gaussian standard deviation = sqrt(2)*alpha).
Energy is multiplied by the SQUARE of each multiplier before shell binning.
No downsampling or spectrum smoothing is applied. Requires numpy/matplotlib.
"""

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path,
                        default=Path(__file__).resolve().parents[1]
                        / "fields/fields_HW_run0_time501_N256.npz")
    parser.add_argument("--lx", type=float, default=50.176)
    parser.add_argument("--ly", type=float, default=50.176,
                        help="Second in-plane length (BOUT z direction)")
    parser.add_argument("--alpha", type=float,
                        help="Filter length in physical units (default: lx/nx)")
    parser.add_argument("--output", type=Path,
                        help="Output image (default: beside input, *_filter_spectra.png)")
    parser.add_argument("--show", action="store_true", help="Also display the figure")
    args = parser.parse_args()
    if not all(np.isfinite(v) and v > 0 for v in (args.lx, args.ly)):
        parser.error("Domain lengths must be finite and positive")

    with np.load(args.input, allow_pickle=False) as data:
        density, potential = (np.asarray(data[key], dtype=float) for key in ("U", "V"))
        time = data["t"].item() if "t" in data else "unknown"
    if (density.ndim != 2 or density.shape != potential.shape
            or min(density.shape) < 2
            or not all(np.isfinite(f).all() for f in (density, potential))):
        parser.error("U and V must be finite, matching 2D fields")
    nx, ny = density.shape
    alpha = args.alpha if args.alpha is not None else args.lx / nx
    if not np.isfinite(alpha) or alpha <= 0:
        parser.error("--alpha must be finite and positive")

    kx = 2 * np.pi * np.fft.fftfreq(nx, d=args.lx / nx)
    ky = 2 * np.pi * np.fft.fftfreq(ny, d=args.ly / ny)
    k2 = kx[:, None] ** 2 + ky[None, :] ** 2
    energy = 0.5 * (np.abs(np.fft.fft2(density)) ** 2
                    + k2 * np.abs(np.fft.fft2(potential)) ** 2) / (nx * ny) ** 2
    dk = min(2 * np.pi / args.lx, 2 * np.pi / args.ly)
    shells = np.rint(np.sqrt(k2) / dk).astype(int)
    multipliers = {
        "Unfiltered": np.ones_like(k2),
        "Gaussian": np.exp(-alpha ** 2 * k2),
        "Germano": 1 / (1 + alpha ** 2 * k2),
    }

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for (label, transfer), style in zip(multipliers.items(), ("k-", "C0--", "C1-.")):
        spectrum = np.bincount(shells.ravel(), weights=(energy * transfer ** 2).ravel())
        k = np.arange(spectrum.size) * dk
        valid = (k > 0) & (spectrum > 0)
        ax.loglog(k[valid], spectrum[valid], style, label=label, linewidth=1.7)
        print(f"{label:10s} total energy (including k=0): {spectrum.sum():.8g}")
    ax.set(xlabel=r"Wavenumber $k$", ylabel=r"energy $E(k)$",
           title=f"HW filter comparison")
    ax.grid(True, which="both", alpha=0.5)
    ax.legend()
    output = args.output or args.input.with_name(args.input.stem + "_filter_spectra.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    ax.set_ylim(top=10)
    ax.set_ylim(bottom=1e-8)
    fig.savefig(output, dpi=180)
    print(f"Saved {output}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
