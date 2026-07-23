#!/usr/bin/env python3
"""Compare HW integral diagnostics and domain-center values versus time.

NPZ snapshots must contain U/V/P (density/potential/vorticity) and simtime.
For a BOUT++ dump, the equivalent variable names n/phi/omega are expected.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
HW_DIR = (
    HERE.parent
    / "BOUT_repo/build_release/examples/hasegawa-wakatani"
)
DEFAULT_DNS = HW_DIR / "results_U-Net_DNS"
DEFAULT_RECONSTRUCTED = HW_DIR / "results_U-Net_32x32"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dns", type=Path, default=DEFAULT_DNS)
    parser.add_argument(
        "--reconstructed", type=Path, default=DEFAULT_RECONSTRUCTED
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("plot_timeseries_comparison.png"),
    )
    parser.add_argument("--lx", type=float, default=50.176)
    parser.add_argument("--ly", type=float, default=50.176)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def _spatial_plane(values: np.ndarray, name: str) -> np.ndarray:
    """Reduce singleton BOUT dimensions, retaining (time, x, y)."""
    values = np.asarray(values)
    if values.ndim < 3:
        raise ValueError(f"{name} must include time and two spatial dimensions")
    # BOUT fields commonly have a singleton y dimension: (t, x, 1, z).
    spatial = list(values.shape[1:])
    squeeze_axes = tuple(
        i + 1 for i, length in enumerate(spatial) if length == 1
    )
    values = np.squeeze(values, axis=squeeze_axes)
    if values.ndim != 3:
        raise ValueError(
            f"{name} has unsupported shape {values.shape}; expected (t,x,y)"
        )
    return values


def _read_npz_directory(path: Path) -> tuple[np.ndarray, ...] | None:
    candidates = sorted(path.rglob("fields_DNS_*.npz"))
    if not candidates:
        candidates = sorted(path.glob("*.npz"))
    records = []
    for filename in candidates:
        with np.load(filename) as source:
            aliases = (
                ("U", "V", "P"),
                ("n", "phi", "omega"),
            )
            names = next(
                (group for group in aliases if all(k in source for k in group)),
                None,
            )
            if names is None:
                continue
            time = float(source["simtime"]) if "simtime" in source else np.nan
            fields = tuple(np.asarray(source[k], dtype=float) for k in names)
        if any(field.ndim != 2 for field in fields):
            raise ValueError(f"{filename}: fields must be two-dimensional")
        records.append((time, *fields))
    if not records:
        return None
    records.sort(key=lambda item: item[0])
    return tuple(np.asarray(items) for items in zip(*records))


def _read_bout_dump(path: Path) -> tuple[np.ndarray, ...]:
    try:
        from netCDF4 import Dataset
    except ImportError as error:
        raise RuntimeError(
            "Reading BOUT.dmp files requires netCDF4 "
            "(install it with `python -m pip install netCDF4`)."
        ) from error

    with Dataset(path) as source:
        def variable(*names: str) -> np.ndarray:
            name = next((name for name in names if name in source.variables), None)
            if name is None:
                raise KeyError(f"{path}: none of {names} is present")
            return np.asarray(source.variables[name][:])

        time = np.ravel(variable("t_array", "t"))
        n = _spatial_plane(variable("n", "U"), "n")
        phi = _spatial_plane(variable("phi", "V"), "phi")
        omega = _spatial_plane(variable("omega", "vort", "P"), "omega")
    if not (len(time) == len(n) == len(phi) == len(omega)):
        raise ValueError(f"{path}: field and time dimensions do not agree")
    return time, n, phi, omega


def read_series(path: Path) -> tuple[np.ndarray, ...]:
    path = path.expanduser().resolve()
    if path.is_file():
        if path.suffix == ".npz":
            result = _read_npz_directory(path.parent)
            if result is not None:
                return result
        return _read_bout_dump(path)
    if not path.is_dir():
        raise FileNotFoundError(path)

    result = _read_npz_directory(path)
    if result is not None:
        return result

    dumps = sorted(path.glob("BOUT.dmp.*.nc"))
    if dumps:
        if len(dumps) != 1:
            raise ValueError(
                f"{path}: multiple processor dumps need to be collected first"
            )
        return _read_bout_dump(dumps[0])

    raise FileNotFoundError(
        f"No U/V/P snapshots or BOUT.dmp.*.nc fields found in {path}"
    )


def diagnostics(
    series: tuple[np.ndarray, ...], lx: float, ly: float
) -> tuple[np.ndarray, ...]:
    time, n, phi, omega = series
    if n.shape != phi.shape or n.shape != omega.shape:
        raise ValueError("n, phi, and omega arrays must have matching shapes")
    nx, ny = n.shape[-2:]
    dx, dy = lx / nx, ly / ny
    dphi_dx = (np.roll(phi, -1, axis=1) - np.roll(phi, 1, axis=1)) / (2 * dx)
    dphi_dy = (np.roll(phi, -1, axis=2) - np.roll(phi, 1, axis=2)) / (2 * dy)
    energy = 0.5 * np.mean(n**2 + dphi_dx**2 + dphi_dy**2, axis=(1, 2))
    enstrophy = 0.5 * np.mean((n - omega) ** 2, axis=(1, 2))
    center = (nx // 2, ny // 2)
    return (
        np.asarray(time),
        energy,
        enstrophy,
        n[:, center[0], center[1]],
        phi[:, center[0], center[1]],
        omega[:, center[0], center[1]],
    )


def main() -> None:
    args = arguments()
    if args.lx <= 0 or args.ly <= 0:
        raise ValueError("lx and ly must be positive")

    datasets = {
        "DNS": diagnostics(read_series(args.dns), args.lx, args.ly),
        "Reconstructed DNS": diagnostics(
            read_series(args.reconstructed),
            args.lx,
            args.ly,
        ),
    }
    quantities = (
        (
            "HW energy",
            r"$E=\frac{1}{2}\langle n^2+|\nabla\phi|^2\rangle$",
        ),
        (
            "HW enstrophy",
            r"$W=\frac{1}{2}\langle(n-\omega)^2\rangle$",
        ),
        ("center density", r"$n(x_c,y_c)$"),
        ("center potential", r"$\phi(x_c,y_c)$"),
        ("center vorticity", r"$\omega(x_c,y_c)$"),
    )
    figure, axes = plt.subplots(3, 2, figsize=(12, 11), sharex=True)
    axes = axes.ravel()
    colors = ("black", "tab:blue")
    for quantity_index, (title, ylabel) in enumerate(quantities, start=1):
        axis = axes[quantity_index - 1]
        for color, (label, values) in zip(colors, datasets.items()):
            axis.plot(values[0], values[quantity_index], color=color, label=label)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
    axes[4].set_xlabel("simulation time")
    axes[0].legend()
    axes[5].axis("off")
    figure.suptitle("Hasegawa–Wakatani DNS comparison")
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
