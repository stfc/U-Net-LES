#!/usr/bin/env python3
"""Animate reconstructed DNS fields and their horizontal centerlines."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, FuncAnimation, PillowWriter


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("results_Unet_32x32/fields"),
        help="Directory containing fields_DNS_*.npz files",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("dns_reconstruction.gif")
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=Path("dns_reconstruction_frames"),
        help="Directory for individual PNG frames",
    )
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument(
        "--limits",
        type=float,
        nargs=6,
        metavar=("U_MIN", "U_MAX", "V_MIN", "V_MAX", "P_MIN", "P_MAX"),
        help=(
            "Fixed min/max limits for density, potential, and vorticity. "
            "By default each field is scaled automatically across all frames."
        ),
    )
    parser.add_argument(
        "--center-index",
        type=int,
        help="Second-axis index for the centerline (default: Ny/2)",
    )
    parser.add_argument("--dpi", type=int, default=120)
    return parser.parse_args()


def read_snapshot(path):
    with np.load(path) as data:
        fields = [np.asarray(data[name]) for name in ("U", "V", "P")]
        step = int(data["pStep"])
        time = float(data["simtime"])
    if any(field.ndim != 2 for field in fields):
        raise ValueError(f"{path}: expected two-dimensional U, V, and P fields")
    return fields, step, time


def main():
    args = parse_args()
    files = sorted(args.input_dir.glob("fields_DNS_*.npz"))
    if not files:
        raise FileNotFoundError(
            f"No fields_DNS_*.npz files found in {args.input_dir}"
        )

    snapshots = [read_snapshot(path) for path in files]
    shape = snapshots[0][0][0].shape
    if any(field.shape != shape for snapshot, _, _ in snapshots for field in snapshot):
        raise ValueError("All fields and snapshots must have the same shape")

    center = shape[1] // 2 if args.center_index is None else args.center_index
    if not 0 <= center < shape[1]:
        raise ValueError(f"center-index must be between 0 and {shape[1] - 1}")

    # Fixed limits avoid distracting rescaling and colorbar changes per frame.
    limits = []
    line_limits = []
    fixed_limits = None
    if args.limits is not None:
        fixed_limits = [tuple(args.limits[i : i + 2]) for i in range(0, 6, 2)]
        if any(lower >= upper for lower, upper in fixed_limits):
            raise ValueError("Each fixed minimum must be smaller than its maximum")
    for channel in range(3):
        if fixed_limits is not None:
            limits.append(fixed_limits[channel])
            line_limits.append(fixed_limits[channel])
            continue
        maximum = max(
            float(np.max(np.abs(fields[channel])))
            for fields, _, _ in snapshots
        )
        maximum = maximum or 1.0
        limits.append((-maximum, maximum))
        line_maximum = max(
            float(np.max(np.abs(fields[channel][:, center])))
            for fields, _, _ in snapshots
        )
        line_maximum = line_maximum or 1.0
        line_limits.append((-1.05 * line_maximum, 1.05 * line_maximum))

    labels = (r"$n$", r"$\phi$", r"$\zeta$")
    cmaps = ("Blues", "Reds_r", "hot")
    figure, axes = plt.subplots(2, 3, figsize=(15, 7.2), constrained_layout=True)
    images = []
    lines = []
    x = np.arange(shape[0])
    initial_fields, initial_step, initial_time = snapshots[0]

    for channel, (label, cmap) in enumerate(zip(labels, cmaps)):
        vmin, vmax = limits[channel]
        image = axes[0, channel].imshow(
            initial_fields[channel].T,
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
            aspect="equal",
        )
        axes[0, channel].set_title(label)
        axes[0, channel].set_xlabel("x index")
        axes[0, channel].set_ylabel("y index")
        figure.colorbar(image, ax=axes[0, channel], shrink=0.9)
        images.append(image)

        (line,) = axes[1, channel].plot(
            x, initial_fields[channel][:, center], color="#8c241c"
        )
        axes[1, channel].set_title(f"{label} at y index {center}")
        axes[1, channel].set_xlim(0, shape[0] - 1)
        axes[1, channel].set_ylim(*line_limits[channel])
        axes[1, channel].set_xlabel("x index")
        axes[1, channel].grid(alpha=0.25)
        lines.append(line)

    title = figure.suptitle(
        f"Reconstructed DNS: step {initial_step}, time {initial_time:.6g}"
    )

    def update(frame):
        fields, step, time = snapshots[frame]
        for channel in range(3):
            images[channel].set_data(fields[channel].T)
            lines[channel].set_ydata(fields[channel][:, center])
        title.set_text(f"Reconstructed DNS: step {step}, time {time:.6g}")
        return [*images, *lines, title]

    args.frames_dir.mkdir(parents=True, exist_ok=True)
    for frame, (_, step, _) in enumerate(snapshots):
        update(frame)
        figure.savefig(
            args.frames_dir / f"fields_DNS_{step:07d}.png", dpi=args.dpi
        )

    animation = FuncAnimation(
        figure, update, frames=len(snapshots), interval=1000.0 / args.fps
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".mp4":
        writer = FFMpegWriter(fps=args.fps)
    else:
        writer = PillowWriter(fps=args.fps)
    animation.save(args.output, writer=writer, dpi=args.dpi)
    plt.close(figure)
    print(
        f"Saved {len(files)} PNG frames to {args.frames_dir} "
        f"and animation to {args.output}"
    )


if __name__ == "__main__":
    main()
