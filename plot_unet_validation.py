#!/usr/bin/env python3
"""Render ``unet_validation_XXXXXXX.npz`` integration diagnostics as images."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


FIELD_NAMES = ("U", "V", "P")
REQUIRED_ARRAYS = (
    "incoming_filtered_fine",
    "downsampled_coarse",
    "unet_filtered_fine",
    "inverse_only_reference_dns",
    "unet_reconstructed_dns",
    "filtered_relative_l2",
    "dns_relative_l2",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot U-Net validation NPZ files produced by pBOUT-unet."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        default=Path("results_U-Net/validation"),
        help=(
            "Validation directory or one unet_validation NPZ file "
            "(default: results_U-Net/validation)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Image directory (default: INPUT/images)",
    )
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--format",
        choices=("png", "pdf"),
        default="png",
        help="Output image format (default: png)",
    )
    parser.add_argument(
        "--lx",
        type=float,
        help="Domain length Lx (overrides archive metadata)",
    )
    parser.add_argument(
        "--ly",
        type=float,
        help="Domain length Ly (overrides archive metadata)",
    )
    return parser.parse_args()


def validation_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Validation input does not exist: {path}")
    files = sorted(path.glob("unet_validation_*.npz"))
    if not files:
        raise FileNotFoundError(
            f"No unet_validation_*.npz files found in {path}"
        )
    return files


def load_validation(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as archive:
        missing = [name for name in REQUIRED_ARRAYS if name not in archive]
        if missing:
            raise KeyError(f"{path} is missing arrays: {missing}")
        data = {name: np.asarray(archive[name]) for name in REQUIRED_ARRAYS}
        data["pStep"] = np.asarray(archive.get("pStep", -1)).item()
        data["simtime"] = np.asarray(archive.get("simtime", np.nan)).item()
        data["lx"] = np.asarray(archive.get("lx", np.nan)).item()
        data["ly"] = np.asarray(archive.get("ly", np.nan)).item()

    fine_shape = data["incoming_filtered_fine"].shape
    if len(fine_shape) != 4 or fine_shape[1] != len(FIELD_NAMES):
        raise ValueError(
            f"{path}: expected fine fields shaped (batch, 3, nx, ny), "
            f"received {fine_shape}"
        )
    for name in (
        "unet_filtered_fine",
        "inverse_only_reference_dns",
        "unet_reconstructed_dns",
    ):
        if data[name].shape != fine_shape:
            raise ValueError(
                f"{path}: {name} has shape {data[name].shape}, "
                f"expected {fine_shape}"
            )
    coarse_shape = data["downsampled_coarse"].shape
    if coarse_shape[:2] != fine_shape[:2] or len(coarse_shape) != 4:
        raise ValueError(
            f"{path}: invalid downsampled_coarse shape {coarse_shape}"
        )
    metric_shape = fine_shape[:2]
    for name in ("filtered_relative_l2", "dns_relative_l2"):
        if data[name].shape != metric_shape:
            raise ValueError(
                f"{path}: {name} has shape {data[name].shape}, "
                f"expected {metric_shape}"
            )
    return data


def symmetric_limit(*arrays: np.ndarray) -> float:
    limit = max(float(np.max(np.abs(array))) for array in arrays)
    return limit if limit > 0 else 1.0


def periodic_derivative(
    field: np.ndarray, spacing: float, axis: int, order: int
) -> np.ndarray:
    """Return a centered periodic first derivative."""
    if order == 2:
        return (
            np.roll(field, -1, axis=axis)
            - np.roll(field, 1, axis=axis)
        ) / (2 * spacing)
    if order == 4:
        if field.shape[axis] < 5:
            raise ValueError(
                "Fourth-order derivative requires at least 5 points"
            )
        return (
            np.roll(field, 2, axis=axis)
            - 8 * np.roll(field, 1, axis=axis)
            + 8 * np.roll(field, -1, axis=axis)
            - np.roll(field, -2, axis=axis)
        ) / (12 * spacing)
    raise ValueError(f"Derivative order must be 2 or 4, received {order}")


def periodic_location_shift(
    first: tuple[int, int],
    second: tuple[int, int],
    shape: tuple[int, int],
    lx: float,
    ly: float,
) -> float:
    """Return the physical separation of two indices on a periodic domain."""
    delta_x = min(
        abs(first[0] - second[0]),
        shape[0] - abs(first[0] - second[0]),
    )
    delta_y = min(
        abs(first[1] - second[1]),
        shape[1] - abs(first[1] - second[1]),
    )
    return float(
        np.hypot(delta_x * lx / shape[0], delta_y * ly / shape[1])
    )


def vorticity_diagnostics(
    reference_w: np.ndarray,
    reconstructed_w: np.ndarray,
    reconstructed_potential: np.ndarray,
    lx: float,
    ly: float,
    order: int,
) -> str:
    """Summarize circulation, extrema/location, and divergence checks."""
    nx, ny = reconstructed_w.shape
    dx, dy = lx / nx, ly / ny

    def circulation(field: np.ndarray) -> tuple[float, float, float]:
        area = dx * dy
        return (
            float(np.sum(field) * area),
            float(np.sum(np.maximum(field, 0)) * area),
            float(np.sum(np.minimum(field, 0)) * area),
        )

    ref_circ = circulation(reference_w)
    rec_circ = circulation(reconstructed_w)
    ref_min = np.unravel_index(np.argmin(reference_w), reference_w.shape)
    ref_max = np.unravel_index(np.argmax(reference_w), reference_w.shape)
    rec_min = np.unravel_index(
        np.argmin(reconstructed_w), reconstructed_w.shape
    )
    rec_max = np.unravel_index(
        np.argmax(reconstructed_w), reconstructed_w.shape
    )
    min_shift = periodic_location_shift(
        ref_min, rec_min, reconstructed_w.shape, lx, ly
    )
    max_shift = periodic_location_shift(
        ref_max, rec_max, reconstructed_w.shape, lx, ly
    )

    # V is electrostatic potential. Its E×B velocity is
    # (-dV/dy, dV/dx), which should be divergence-free.
    velocity_x = -periodic_derivative(
        reconstructed_potential, dy, axis=1, order=order
    )
    velocity_y = periodic_derivative(
        reconstructed_potential, dx, axis=0, order=order
    )
    divergence = periodic_derivative(
        velocity_x, dx, axis=0, order=order
    ) + periodic_derivative(velocity_y, dy, axis=1, order=order)

    return (
        "Circulation Γ (total / positive / negative)\n"
        f"ref: {ref_circ[0]:+.3e} / {ref_circ[1]:+.3e} / "
        f"{ref_circ[2]:+.3e}\n"
        f"rec: {rec_circ[0]:+.3e} / {rec_circ[1]:+.3e} / "
        f"{rec_circ[2]:+.3e}\n\n"
        "Extrema: value @ grid index (ix, iy)\n"
        f"min ref: {reference_w[ref_min]:+.3e} @ {ref_min}\n"
        f"min rec: {reconstructed_w[rec_min]:+.3e} @ {rec_min}\n"
        f"max ref: {reference_w[ref_max]:+.3e} @ {ref_max}\n"
        f"max rec: {reconstructed_w[rec_max]:+.3e} @ {rec_max}\n"
        f"periodic shift min/max: {min_shift:.3e} / {max_shift:.3e}\n\n"
        "div(E×B velocity), reconstructed V\n"
        f"RMS: {np.sqrt(np.mean(divergence**2)):.3e}   "
        f"max|.|: {np.max(np.abs(divergence)):.3e}"
    )


def plot_field(
    data: dict[str, np.ndarray],
    source: Path,
    output_dir: Path,
    batch: int,
    channel: int,
    dpi: int,
    image_format: str,
    lx: float,
    ly: float,
) -> Path:
    incoming = data["incoming_filtered_fine"][batch, channel]
    coarse = data["downsampled_coarse"][batch, channel]
    predicted = data["unet_filtered_fine"][batch, channel]
    reference_dns = data["inverse_only_reference_dns"][batch, channel]
    reconstructed_dns = data["unet_reconstructed_dns"][batch, channel]
    filtered_error = predicted - incoming
    dns_error = reconstructed_dns - reference_dns

    filtered_limit = symmetric_limit(incoming, predicted)
    dns_limit = symmetric_limit(reference_dns, reconstructed_dns)
    filtered_error_limit = symmetric_limit(filtered_error)
    dns_error_limit = symmetric_limit(dns_error)

    panels = (
        (incoming, "Incoming filtered fine", filtered_limit),
        (coarse, "Spectrally downsampled", symmetric_limit(coarse)),
        (predicted, "U-Net filtered fine", filtered_limit),
        (
            filtered_error,
            "U-Net filtered − incoming",
            filtered_error_limit,
        ),
        (reference_dns, "Inverse-only DNS reference", dns_limit),
        (reconstructed_dns, "U-Net reconstructed DNS", dns_limit),
        (
            dns_error,
            "Reconstructed − reference DNS",
            dns_error_limit,
        ),
    )

    fig, axes = plt.subplots(
        2,
        4,
        figsize=(17, 4.25 * rows),
        constrained_layout=True,
    )
    axes = axes.ravel()
    extent = (0, 1, 0, 1)
    for axis, (field, title, limit) in zip(axes, panels):
        image = axis.imshow(
            field.T,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
            interpolation="nearest",
            aspect="equal",
        )
        axis.set_title(title)
        axis.set_xlabel("x / Lx")
        axis.set_ylabel("z / Lz")
        fig.colorbar(image, ax=axis, shrink=0.82)

    filtered_l2 = data["filtered_relative_l2"][batch, channel]
    dns_l2 = data["dns_relative_l2"][batch, channel]
    axes[7].axis("off")
    axes[7].text(
        0.05,
        0.75,
        f"Filtered relative L2: {filtered_l2:.6e}\n"
        f"DNS relative L2: {dns_l2:.6e}\n\n"
        f"Fine grid: {incoming.shape[0]} × {incoming.shape[1]}\n"
        f"Coarse grid: {coarse.shape[0]} × {coarse.shape[1]}",
        va="top",
        family="monospace",
        fontsize=11,
    )
    fig.suptitle(
        f"{FIELD_NAMES[channel]} reconstruction — step {data['pStep']}, "
        f"t={data['simtime']:g}, batch {batch}"
    )

    batch_suffix = (
        "" if data["incoming_filtered_fine"].shape[0] == 1
        else f"_batch{batch:03d}"
    )
    output_path = output_dir / (
        f"{source.stem}_{FIELD_NAMES[channel]}{batch_suffix}.{image_format}"
    )
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def main():
    args = parse_args()
    files = validation_files(args.input)
    default_parent = args.input if args.input.is_dir() else args.input.parent
    output_dir = args.output_dir or default_parent / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    images_written = 0
    for index, source in enumerate(files, 1):
        data = load_validation(source)
        lx = args.lx if args.lx is not None else float(data["lx"])
        ly = args.ly if args.ly is not None else float(data["ly"])
        if not np.isfinite(lx) or lx <= 0 or not np.isfinite(ly) or ly <= 0:
            raise ValueError(
                f"{source}: missing valid lx/ly metadata; provide --lx and --ly"
            )
        for batch in range(data["incoming_filtered_fine"].shape[0]):
            for channel in range(len(FIELD_NAMES)):
                plot_field(
                    data,
                    source,
                    output_dir,
                    batch,
                    channel,
                    args.dpi,
                    args.format,
                    lx,
                    ly,
                )
                images_written += 1
        print(f"[{index}/{len(files)}] {source}")

    print(f"Saved {images_written} images to {output_dir}")


if __name__ == "__main__":
    main()
