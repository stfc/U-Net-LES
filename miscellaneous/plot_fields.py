#!/usr/bin/env python3
"""Plot an NPZ snapshot and report wraparound smoothness (requires numpy/matplotlib)."""

import argparse
from pathlib import Path

import numpy as np


def rms(values):
    return float(np.sqrt(np.mean(np.square(values))))


def ratio(numerator, denominator):
    return numerator / denominator if denominator else (0.0 if numerator == 0 else float("inf"))


def periodic_derivative(values, order, spacing):
    """Fourier derivative of a centerline, assuming periodic samples."""
    wave = 2 * np.pi * np.fft.fftfreq(values.size, d=spacing)
    multiplier = (1j * wave) ** order
    # Odd derivatives of the real Nyquist mode vanish on an even-sized grid.
    if values.size % 2 == 0 and order % 2:
        multiplier[values.size // 2] = 0
    return np.fft.ifft(multiplier * np.fft.fft(values)).real


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=Path(__file__).resolve().parent
                        / "fields/fields_mHW_run0_time144_N256.npz")
    parser.add_argument("--output", type=Path, help="PNG path (default: beside the NPZ)")
    parser.add_argument("--show", action="store_true", help="Also open an interactive figure")
    parser.add_argument("--dx", type=float, default=1.0, help="x spacing (default: 1 index unit)")
    parser.add_argument("--dy", type=float, default=1.0, help="y spacing (default: 1 index unit)")
    args = parser.parse_args()
    if not all(np.isfinite(s) and s > 0 for s in (args.dx, args.dy)):
        parser.error("--dx and --dy must be finite and positive")

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with np.load(args.input, allow_pickle=False) as data:
        fields = {key: np.asarray(data[key], dtype=float) for key in ("U", "V", "P")}
        time = data["t"].item() if "t" in data else "unknown"
    for key, field in fields.items():
        if field.ndim != 2 or min(field.shape) < 3 or not np.isfinite(field).all():
            raise ValueError(f"{key} must be a finite 2D field with at least 3 samples per axis")

    # The repository stores fields as (x, y). No physical coordinates are in this NPZ.
    fig, axes = plt.subplots(9, 3, figsize=(15, 27), constrained_layout=True,
                             gridspec_kw={"height_ratios": [2.0] + [1.0] * 8})
    print("Axes: x=axis 0, y=axis 1. RMS ratios compare seam to interior differences.")
    print(f"{'Field':<7} {'Axis':<5} {'Seam RMS':>12} {'Interior RMS':>14} {'Step ratio':>12} {'Curvature ratio':>17}")
    for col, (key, field) in enumerate(fields.items()):
        ax = axes[0, col]
        scale = float(np.max(np.abs(field))) or 1.0
        im = ax.imshow(field.T, origin="lower", cmap="RdBu_r", vmin=-scale, vmax=scale)
        ax.set(title=key, xlabel="x index", ylabel="y index")
        nx, ny = field.shape
        ax.axhline(ny // 2, color="black", linestyle="--", linewidth=0.8)
        ax.axvline(nx // 2, color="black", linestyle=":", linewidth=0.8)
        fig.colorbar(im, ax=ax, shrink=0.85)
        for order in range(1, 5):
            for direction, (name, line, spacing, center) in enumerate((
                ("x", field[:, ny // 2], args.dx, f"y index = {ny // 2}"),
                ("y", field[nx // 2, :], args.dy, f"x index = {nx // 2}"),
            )):
                derivative_ax = axes[1 + 2 * (order - 1) + direction, col]
                coordinate = np.arange(line.size) * spacing
                derivative_ax.plot(coordinate, periodic_derivative(line, order, spacing),
                                   color="tab:blue" if direction == 0 else "tab:orange",
                                   linewidth=1)
                derivative_ax.set(
                    title=rf"$\partial^{order} {key}/\partial {name}^{order}$ | {center}",
                    xlabel=f"{name} index" if spacing == 1 else f"{name} (spacing {spacing:g})",
                    ylabel="Derivative",
                    xlim=(coordinate[0], coordinate[-1]),
                )
                derivative_ax.axhline(0, color="gray", linewidth=0.5)
                derivative_ax.grid(alpha=0.25)
                derivative_ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 3))
        for axis, name in enumerate(("x", "y")):
            a = np.moveaxis(field, axis, 0)
            seam = rms(a[0] - a[-1])
            interior = rms(np.diff(a, axis=0))
            seam_curvature = rms(np.stack((a[1] - 2*a[0] + a[-1],
                                          a[0] - 2*a[-1] + a[-2])))
            interior_curvature = rms(np.diff(a, n=2, axis=0))
            print(f"{key:<7} {name:<5} {seam:12.5g} {interior:14.5g} "
                  f"{ratio(seam, interior):12.3f} {ratio(seam_curvature, interior_curvature):17.3f}")
    print("Ratios near unity or smaller support a smooth periodic join; large ratios flag a seam.")
    print("First/last samples need not match on a periodic grid without a duplicated endpoint.")
    print("A snapshot alone cannot establish the simulation boundary conditions.")
    fig.suptitle(f"{args.input.name} | t = {time}\n"
                 f"Centerline Fourier derivatives (periodic assumption); dx={args.dx:g}, dy={args.dy:g}")
    output = args.output or args.input.with_suffix(".png")
    fig.savefig(output, dpi=180)
    print(f"Saved {output}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
