#!/usr/bin/env python3
"""Compare U-Net and DNS field-snapshot histories.

Read ``fields_*.npz`` snapshots from two specified directories, then plot
domain-center values and total/component HW-energy histories.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from visualization_common import (
    HW_FIELD_LABELS,
    calculate_arakawa_terms,
    save_png_animation,
)


FIELD_NAMES = ("U", "V", "P")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "unet_fields",
        type=Path,
        help="First run directory, or its directory of fields_*.npz snapshots",
    )
    parser.add_argument(
        "dns_fields",
        type=Path,
        help="Second run directory, or its directory of fields_*.npz snapshots",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results_U-Net/comparison"),
        help="Output directory (default: results_U-Net/comparison)",
    )
    parser.add_argument("--lx", type=float, default=50.176)
    parser.add_argument("--ly", type=float, default=50.176)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--fps", type=float, default=5.0,
        help="Frame rate of the 2-D comparison animation (default: 5)",
    )
    parser.add_argument(
        "--animations",
        action="store_true",
        help="Generate field and Arakawa comparison GIFs (disabled by default)",
    )
    parser.add_argument(
        "--batch", type=int, default=0,
        help="Batch index for batched snapshots (default: 0)",
    )
    return parser.parse_args()


def _select_field(values: np.ndarray, batch: int, source: Path) -> np.ndarray:
    """Select a 2-D field from an unbatched or batched archive array."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 2:
        if batch != 0:
            raise IndexError(f"{source}: unbatched field has no batch {batch}")
        return values
    if values.ndim == 3:
        if not 0 <= batch < values.shape[0]:
            raise IndexError(
                f"{source}: batch {batch} is outside [0, {values.shape[0]})"
            )
        return values[batch]
    raise ValueError(
        f"{source}: expected (nx, ny) or (batch, nx, ny), got {values.shape}"
    )


def read_history(
    directory: Path, batch: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read time, step, and channel-first fields from a snapshot directory."""
    if not directory.is_dir():
        raise FileNotFoundError(f"Field directory does not exist: {directory}")
    nested_fields = directory / "fields"
    fields_directory = nested_fields if nested_fields.is_dir() else directory
    files = sorted(fields_directory.glob("fields_*.npz"))
    if not files:
        raise FileNotFoundError(
            f"No fields_*.npz files found in {directory} or "
            f"{directory / 'fields'}"
        )

    records = []
    expected_shape = None
    for source in files:
        with np.load(source) as archive:
            missing = [name for name in FIELD_NAMES if name not in archive]
            if missing:
                raise KeyError(f"{source} is missing fields: {missing}")
            fields = np.stack(
                [_select_field(archive[name], batch, source) for name in FIELD_NAMES]
            )
            simtime = float(np.asarray(archive.get("simtime", np.nan)).item())
            step = int(np.asarray(archive.get("pStep", -1)).item())
        if not np.isfinite(simtime):
            raise ValueError(f"{source}: missing a finite simtime")
        if expected_shape is None:
            expected_shape = fields.shape
        elif fields.shape != expected_shape:
            raise ValueError(
                f"{source}: field shape {fields.shape} differs from {expected_shape}"
            )
        records.append((simtime, step, fields))

    records.sort(key=lambda record: (record[0], record[1]))
    return (
        np.asarray([record[0] for record in records]),
        np.asarray([record[1] for record in records]),
        np.stack([record[2] for record in records]),
    )


def periodic_derivative(
    values: np.ndarray, spacing: float, axis: int
) -> np.ndarray:
    """Return a fourth-order centered derivative on a periodic grid."""
    if values.shape[axis] < 5:
        raise ValueError("Fourth-order derivative requires at least five points")
    return (
        -np.roll(values, -2, axis=axis)
        + 8.0 * np.roll(values, -1, axis=axis)
        - 8.0 * np.roll(values, 1, axis=axis)
        + np.roll(values, 2, axis=axis)
    ) / (12.0 * spacing)


def hw_energy_components(
    fields: np.ndarray, lx: float, ly: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the density, x-gradient, and y-gradient HW energy terms."""
    if min(fields.shape[-2:]) < 5:
        raise ValueError("HW energy requires at least five points per direction")
    density, potential = fields[:, 0], fields[:, 1]
    dx, dy = lx / potential.shape[-2], ly / potential.shape[-1]
    dphi_dx = periodic_derivative(potential, dx, -2)
    dphi_dy = periodic_derivative(potential, dy, -1)
    return (
        0.5 * np.mean(density**2, axis=(-2, -1)),
        0.5 * np.mean(dphi_dx**2, axis=(-2, -1)),
        0.5 * np.mean(dphi_dy**2, axis=(-2, -1)),
    )


def hw_energy(fields: np.ndarray, lx: float, ly: float) -> np.ndarray:
    """Compute E=1/2 <n^2 + |grad(phi)|^2> with periodic 4th-order gradients."""
    return np.sum(hw_energy_components(fields, lx, ly), axis=0)


def save_centerpoint_history(datasets, output_path: Path, dpi: int) -> Path:
    """Compare each field's value at the domain center versus time."""
    figure, axes = plt.subplots(
        len(FIELD_NAMES), 1, figsize=(9, 10), sharex=True, squeeze=False
    )
    styles = (("tab:blue", "-"), ("black", "--"))
    for channel, name in enumerate(FIELD_NAMES):
        axis = axes[channel, 0]
        for (label, times, fields), (color, linestyle) in zip(datasets, styles):
            center_x = fields.shape[-2] // 2
            center_y = fields.shape[-1] // 2
            axis.plot(
                times,
                fields[:, channel, center_x, center_y],
                color=color,
                linestyle=linestyle,
                linewidth=1.6,
                label=label,
            )
        field_label = HW_FIELD_LABELS.get(name, name)
        axis.set_title(f"{field_label} at domain center")
        axis.set_ylabel(field_label)
        axis.grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[-1, 0].set_xlabel("simulation time")
    figure.suptitle("Domain-center field trajectories")
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_energy_comparison(
    datasets, output_path: Path, lx: float, ly: float, dpi: int
) -> Path:
    """Plot overlaid HW energy histories."""
    figure, axis = plt.subplots(figsize=(7.4, 5.2))
    styles = (("tab:blue", "-"), ("black", "--"))
    for (label, times, fields), (color, linestyle) in zip(datasets, styles):
        axis.plot(
            times, hw_energy(fields, lx, ly), color=color,
            linestyle=linestyle, linewidth=1.6, label=label,
        )
    axis.set_xlabel("simulation time")
    axis.set_ylabel(r"$E=\frac{1}{2}\langle n^2+|\nabla\phi|^2\rangle$")
    axis.set_title("Hasegawa–Wakatani energy versus time")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_energy_components_comparison(
    datasets, output_path: Path, lx: float, ly: float, dpi: int
) -> Path:
    """Plot each additive HW energy component versus time."""
    component_labels = (
        r"Density: $E_n=\frac{1}{2}\langle n^2\rangle$",
        r"Potential x-gradient: $E_{\phi,x}=\frac{1}{2}\langle(\partial_x\phi)^2\rangle$",
        r"Potential y-gradient: $E_{\phi,y}=\frac{1}{2}\langle(\partial_y\phi)^2\rangle$",
    )
    figure, axes = plt.subplots(
        len(component_labels), 1, figsize=(8.2, 10), sharex=True,
        squeeze=False,
    )
    styles = (("tab:blue", "-"), ("black", "--"))
    for (label, times, fields), (color, linestyle) in zip(datasets, styles):
        components = hw_energy_components(fields, lx, ly)
        for axis, values in zip(axes[:, 0], components):
            axis.plot(
                times, values, color=color, linestyle=linestyle,
                linewidth=1.6, label=label,
            )

    for axis, component_label in zip(axes[:, 0], component_labels):
        axis.set_title(component_label)
        axis.set_ylabel("Energy")
        axis.grid(True, alpha=0.3)
    axes[0, 0].legend()
    axes[-1, 0].set_xlabel("simulation time")
    figure.suptitle("Hasegawa–Wakatani energy components versus time")
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_initial_derivative_comparison(
    datasets, output_path: Path, lx: float, ly: float, dpi: int
) -> Path:
    """Compare x/y derivatives of every field at the time nearest zero."""
    (first_label, first_times, first_fields), (
        second_label, second_times, second_fields
    ) = datasets
    if first_fields.shape[1:] != second_fields.shape[1:]:
        raise ValueError(
            "Derivative comparison requires matching field grids; received "
            f"{first_fields.shape[2:]} and {second_fields.shape[2:]}"
        )

    first_index = int(np.argmin(np.abs(first_times)))
    second_index = int(np.argmin(np.abs(second_times)))
    nx, ny = first_fields.shape[-2:]
    spacings = (lx / nx, ly / ny)
    directions = (("x", -2), ("y", -1))
    figure, axes = plt.subplots(
        2 * len(FIELD_NAMES), 3, figsize=(14, 24), squeeze=False
    )
    for channel, name in enumerate(FIELD_NAMES):
        field_label = HW_FIELD_LABELS.get(name, name)
        for direction_index, ((direction, axis), spacing) in enumerate(
            zip(directions, spacings)
        ):
            row = 2 * channel + direction_index
            first_values = periodic_derivative(
                first_fields[first_index, channel], spacing, axis
            )
            second_values = periodic_derivative(
                second_fields[second_index, channel], spacing, axis
            )
            error = np.abs(first_values - second_values)
            limit = max(
                float(np.max(np.abs(first_values))),
                float(np.max(np.abs(second_values))),
            ) or 1.0
            error_limit = float(np.max(error)) or 1.0
            derivative_label = rf"$\partial_{direction}$ {field_label}"
            panels = (
                (first_values, first_label, "RdBu_r", -limit, limit),
                (second_values, second_label, "RdBu_r", -limit, limit),
                (error, "absolute error", "magma", 0.0, error_limit),
            )
            for column, (values, label, cmap, vmin, vmax) in enumerate(panels):
                plot_axis = axes[row, column]
                image = plot_axis.imshow(
                    values.T,
                    origin="lower",
                    extent=(0.0, 1.0, 0.0, 1.0),
                    aspect="equal",
                    interpolation="nearest",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )
                plot_axis.set_title(f"{derivative_label}: {label}")
                plot_axis.set_xlabel("x / Lx")
                plot_axis.set_ylabel("y / Ly")
                figure.colorbar(image, ax=plot_axis, shrink=0.78)
    figure.suptitle(
        f"Initial field derivatives: {first_label} t={first_times[first_index]:g}, "
        f"{second_label} t={second_times[second_index]:g}"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_field_comparison_animation(
    datasets, output_dir: Path, dpi: int, fps: float
) -> Path:
    """Animate both 2-D solutions and their absolute error at nearest times."""
    (first_label, first_times, first_fields), (
        second_label, second_times, second_fields
    ) = datasets
    if first_fields.shape[1:] != second_fields.shape[1:]:
        raise ValueError(
            "2-D error animation requires matching field grids; received "
            f"{first_fields.shape[2:]} and {second_fields.shape[2:]}"
        )

    nearest = np.asarray(
        [np.argmin(np.abs(second_times - time)) for time in first_times]
    )
    matched_second = second_fields[nearest]
    errors = np.abs(first_fields - matched_second)
    field_limits = [
        max(
            float(np.max(np.abs(first_fields[:, channel]))),
            float(np.max(np.abs(matched_second[:, channel]))),
        )
        for channel in range(len(FIELD_NAMES))
    ]
    error_limits = [
        float(np.max(errors[:, channel]))
        for channel in range(len(FIELD_NAMES))
    ]

    frames_dir = output_dir / "field_comparison_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []
    for frame_index, (first_time, second_index) in enumerate(
        zip(first_times, nearest)
    ):
        figure, axes = plt.subplots(
            len(FIELD_NAMES), 3, figsize=(14, 13), squeeze=False
        )
        for channel, name in enumerate(FIELD_NAMES):
            limit = field_limits[channel] or 1.0
            error_limit = error_limits[channel] or 1.0
            panels = (
                (first_fields[frame_index, channel], first_label, "RdBu_r", -limit, limit),
                (second_fields[second_index, channel], second_label, "RdBu_r", -limit, limit),
                (errors[frame_index, channel], "absolute error", "magma", 0.0, error_limit),
            )
            for column, (values, label, cmap, vmin, vmax) in enumerate(panels):
                axis = axes[channel, column]
                image = axis.imshow(
                    values.T,
                    origin="lower",
                    extent=(0.0, 1.0, 0.0, 1.0),
                    aspect="equal",
                    interpolation="nearest",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )
                axis.set_title(f"{HW_FIELD_LABELS.get(name, name)}: {label}")
                axis.set_xlabel("x / Lx")
                axis.set_ylabel("y / Ly")
                figure.colorbar(image, ax=axis, shrink=0.78)
        figure.suptitle(
            f"2-D field comparison: {first_label} t={first_time:g}, "
            f"{second_label} t={second_times[second_index]:g}"
        )
        figure.tight_layout()
        frame_path = frames_dir / f"field_comparison_{frame_index:07d}.png"
        figure.savefig(frame_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        frame_paths.append(frame_path)

    return save_png_animation(
        frame_paths, output_dir / "field_comparison.gif", fps=fps
    )


def save_arakawa_comparison_animation(
    datasets, output_dir: Path, lx: float, ly: float, dpi: int, fps: float
) -> Path:
    """Animate U-Net and DNS Arakawa terms and their absolute errors."""
    (first_label, first_times, first_fields), (
        second_label, second_times, second_fields
    ) = datasets
    if first_fields.shape[1:] != second_fields.shape[1:]:
        raise ValueError(
            "Arakawa comparison requires matching field grids; received "
            f"{first_fields.shape[2:]} and {second_fields.shape[2:]}"
        )

    nearest = np.asarray(
        [np.argmin(np.abs(second_times - time)) for time in first_times]
    )
    first_terms = np.asarray(
        [calculate_arakawa_terms(fields, lx, ly) for fields in first_fields]
    )
    second_terms = np.asarray(
        [
            calculate_arakawa_terms(second_fields[index], lx, ly)
            for index in nearest
        ]
    )
    errors = np.abs(first_terms - second_terms)
    term_limits = [
        max(
            float(np.max(np.abs(first_terms[:, term]))),
            float(np.max(np.abs(second_terms[:, term]))),
        )
        for term in range(2)
    ]
    error_limits = [
        float(np.max(errors[:, term])) for term in range(2)
    ]
    term_labels = (r"$J(\phi,\omega)$", r"$J(\phi,n)$")

    frames_dir = output_dir / "arakawa_comparison_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []
    for frame_index, (first_time, second_index) in enumerate(
        zip(first_times, nearest)
    ):
        figure, axes = plt.subplots(2, 3, figsize=(14, 9), squeeze=False)
        for term, term_label in enumerate(term_labels):
            limit = term_limits[term] or 1.0
            error_limit = error_limits[term] or 1.0
            panels = (
                (first_terms[frame_index, term], first_label, "RdBu_r", -limit, limit),
                (second_terms[frame_index, term], second_label, "RdBu_r", -limit, limit),
                (errors[frame_index, term], "absolute error", "magma", 0.0, error_limit),
            )
            for column, (values, label, cmap, vmin, vmax) in enumerate(panels):
                axis = axes[term, column]
                image = axis.imshow(
                    values.T,
                    origin="lower",
                    extent=(0.0, 1.0, 0.0, 1.0),
                    aspect="equal",
                    interpolation="nearest",
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                )
                axis.set_title(f"{term_label}: {label}")
                axis.set_xlabel("x / Lx")
                axis.set_ylabel("y / Ly")
                figure.colorbar(image, ax=axis, shrink=0.78)
        figure.suptitle(
            f"Arakawa comparison: {first_label} t={first_time:g}, "
            f"{second_label} t={second_times[second_index]:g}"
        )
        figure.tight_layout()
        frame_path = frames_dir / f"arakawa_comparison_{frame_index:07d}.png"
        figure.savefig(frame_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        frame_paths.append(frame_path)

    return save_png_animation(
        frame_paths, output_dir / "arakawa_comparison.gif", fps=fps
    )


def main() -> None:
    args = parse_args()
    if args.lx <= 0 or args.ly <= 0:
        raise ValueError("lx and ly must be positive")
    if args.batch < 0:
        raise ValueError("batch must be non-negative")
    if args.animations and args.fps <= 0:
        raise ValueError("fps must be positive")

    unet_times, _, unet_fields = read_history(args.unet_fields, args.batch)
    dns_times, _, dns_fields = read_history(args.dns_fields, args.batch)
    datasets = (
        (args.unet_fields.resolve().name, unet_times, unet_fields),
        (args.dns_fields.resolve().name, dns_times, dns_fields),
    )
    output_directory = args.output_dir.resolve()
    input_directories = (
        args.unet_fields.resolve(),
        args.dns_fields.resolve(),
    )
    if any(
        input_directory == output_directory
        or output_directory in input_directory.parents
        for input_directory in input_directories
    ):
        raise ValueError(
            f"Refusing to delete output directory {output_directory} because "
            "it contains an input dataset"
        )
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    centerpoint_path = save_centerpoint_history(
        datasets, args.output_dir / "centerpoint_vs_time.png", args.dpi
    )
    energy_path = save_energy_comparison(
        datasets, args.output_dir / "energy_vs_time.png",
        args.lx, args.ly, args.dpi,
    )
    energy_components_path = save_energy_components_comparison(
        datasets, args.output_dir / "energy_components_vs_time.png",
        args.lx, args.ly, args.dpi,
    )
    initial_derivatives_path = save_initial_derivative_comparison(
        datasets, args.output_dir / "field_derivatives_t0.png",
        args.lx, args.ly, args.dpi,
    )
    print(f"Saved {centerpoint_path}")
    print(f"Saved {energy_path}")
    print(f"Saved {energy_components_path}")
    print(f"Saved {initial_derivatives_path}")
    if args.animations:
        animation_path = save_field_comparison_animation(
            datasets, args.output_dir, args.dpi, args.fps
        )
        arakawa_animation_path = save_arakawa_comparison_animation(
            datasets, args.output_dir, args.lx, args.ly, args.dpi, args.fps
        )
        print(f"Saved {animation_path}")
        print(f"Saved {arakawa_animation_path}")


if __name__ == "__main__":
    main()
