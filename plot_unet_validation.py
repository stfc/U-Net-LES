#!/usr/bin/env python3
"""Compare up to four field-snapshot histories.

Read ``fields_*.npz`` snapshots or BOUT NetCDF dumps, then plot
domain-center values, total/component HW-energy, enstrophy, and particle-flux
histories. The dataset whose
directory name contains ``DNS`` is used as the reference; if none does, the
first dataset is the reference. Animations and the initial-derivative plot
compare the first non-reference dataset with it.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil

# Set before third-party imports can initialize HDF5 on the shared filesystem.
# Preserve an explicit user override (for example HDF5_USE_FILE_LOCKING=TRUE).
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from visualization_common import \
(
    HW_FIELD_LABELS,
    save_png_animation,
)
from germano_common import \
(
    SPECTRUM_Y_MIN,
    SPECTRUM_Y_MAX,
    isotropic_hw_energy_spectrum,
)


FIELD_NAMES = ("U", "V", "P")
MAX_ANIMATION_FRAMES = 1000
DEFAULT_BOUT_RESULTS_ROOT = \
(
    Path(__file__).resolve().parent
    / "../BOUT_repo/build_release/examples/hasegawa-wakatani"
).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument \
    (
        "unet_fields",
        type=Path,
        help="First run/snapshot directory or BOUT dump file (MPI splits supported)",
    )
    parser.add_argument \
    (
        "dns_fields",
        type=Path,
        nargs="?",
        help="Second run/snapshot directory or BOUT dump file (MPI splits supported)",
    )
    parser.add_argument \
    (
        "dataset2_fields", type=Path, nargs="?",
        help="Optional second model dataset directory"
    )
    parser.add_argument \
    (
        "dataset3_fields", type=Path, nargs="?",
        help="Optional third model dataset directory"
    )
    parser.add_argument \
    (
        "--output-dir", type=Path, default=Path("validation_U-Net/comparison"),
        help="Output directory (default: validation_U-Net/comparison)",
    )
    parser.add_argument \
    (
        "--base-dir",
        type=Path,
        default=DEFAULT_BOUT_RESULTS_ROOT,
        help= \
        (
            "Directory prepended to every relative input dataset path "
            f"(default: {DEFAULT_BOUT_RESULTS_ROOT})"
        ),
    )
    parser.add_argument(
        "--input-format", choices=("auto", "npz", "netcdf"), default="auto",
        help="Input format for all datasets; auto prefers NPZ in directories "
             "and accepts .nc files directly (default: auto)",
    )
    parser.add_argument(
        "--bout-y-index", type=int, default=0,
        help="Interior y-plane for BOUT (t,x,y,z) fields (default: 0)",
    )
    parser.add_argument("--lx", type=float, default=50.176)
    parser.add_argument("--ly", type=float, default=50.176)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument \
    (
        "--fps", type=float, default=5.0,
        help="Frame rate of the 2-D comparison animation (default: 5)",
    )
    parser.add_argument \
    (
        "--animations",
        action="store_true",
        help="Generate field and energy-spectrum comparison GIFs (disabled by default)",
    )
    parser.add_argument \
    (
        "--smooth-spectrum",
        action="store_true",
        help= \
        (
            "Smooth energy spectra with the StylES five-bin moving average; "
            "the first four bins remain unsmoothed"
        ),
    )
    parser.add_argument \
    (
        "--spectrum-cutoff-n",
        type=int,
        default=64,
        help= \
        (
            "Grid size whose Nyquist wavenumber is marked on the static "
            "initial/final HW-spectrum comparison; 0 disables the marker (default: 64)"
        ),
    )
    parser.add_argument \
    (
        "--overall-animation-limits", action="store_true",
        help= \
        (
            "Set animation color limits from all frames instead of the "
            "snapshots nearest t=0"
        ),
    )
    parser.add_argument \
    (
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
            raise IndexError \
            (
                f"{source}: batch {batch} is outside [0, {values.shape[0]})"
            )
        return values[batch]
    raise ValueError \
    (
        f"{source}: expected (nx, ny) or (batch, nx, ny), got {values.shape}"
    )


def resolve_input_directory(path: Path, base_directory: Path) -> Path:
    """Resolve relative dataset paths beneath the shared BOUT++ results root."""
    candidate = path if path.is_absolute() else base_directory / path
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Input dataset does not exist: {candidate}")


def _read_mpi_bout_history(source: Path, y_index: int, nranks: int):
    """Collect sibling MPI dumps, checking completeness before assembly."""
    from netCDF4 import Dataset
    try:
        from boutdata import collect
    except ImportError as error:
        raise ImportError("Reading MPI dumps requires boutdata (pip install boutdata)") from error

    prefix, rank, suffix = source.name.rsplit(".", 2)
    if not rank.isdigit() or suffix != "nc":
        raise ValueError(f"{source}: expected a <prefix>.<rank>.nc MPI dump")
    if y_index < 0:
        raise ValueError(f"{source}: y index must be non-negative")
    times = None
    for rank in range(nranks):
        filename = source.with_name(f"{prefix}.{rank}.nc")
        if not filename.is_file():
            raise FileNotFoundError(f"Missing MPI dump: {filename}")
        with Dataset(filename) as archive:
            missing = [name for name in ("t_array", "n", "phi", "vort")
                       if name not in archive.variables]
            if missing:
                raise KeyError(f"{filename} is missing variables: {missing}")
            rank_times = np.asarray(np.ma.filled(archive["t_array"][:], np.nan))
            if rank_times.ndim != 1 or not rank_times.size or not np.all(np.isfinite(rank_times)):
                raise ValueError(f"{filename}: t_array must be a nonempty finite 1-D array")
            if times is not None and not np.array_equal(times, rank_times):
                raise ValueError(f"{filename}: MPI ranks have inconsistent output times")
            times = rank_times
            for name in ("n", "phi", "vort"):
                variable = archive[name]
                if variable.dimensions != ("t", "x", "y", "z"):
                    raise ValueError(f"{filename}: {name} has unsupported dimensions {variable.dimensions}")
                if variable.shape[0] != times.size:
                    raise ValueError(f"{filename}: invalid time extent for {name}")
            ny = int(np.asarray(archive["MYSUB"][...]).item()) * int(np.asarray(archive["NYPE"][...]).item())
            if y_index >= ny:
                raise IndexError(f"{source}: y index {y_index} outside interior [0, {ny})")

    planes = []
    for name in ("n", "phi", "vort"):
        values = collect(name, path=str(source.parent), prefix=prefix,
                         xguards=False, yguards=False, yind=y_index,
                         info=False, strict=True)
        values = np.asarray(np.ma.filled(values, np.nan), dtype=np.float64)
        if values.ndim != 4 or values.shape[0] != times.size or values.shape[2] != 1:
            raise ValueError(f"{source}: unexpected collected shape for {name}: {values.shape}")
        values = values[:, :, 0, :]
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{source}: {name} contains missing or non-finite values")
        if planes and values.shape != planes[0].shape:
            raise ValueError(f"{source}: inconsistent field shapes")
        planes.append(values)
    order = np.argsort(times, kind="stable")
    return times[order], np.arange(times.size)[order], np.stack(planes, axis=1)[order]


def read_bout_history(source: Path, y_index: int = 0):
    """Read single-rank, MPI-split or squashed HW dumps without guards.

    Map n/phi/vort to U/V/P. Steps are output indices, not solver steps.
    For MPI dumps, assemble sibling rank files with boutdata.
    """
    try:
        from netCDF4 import Dataset
    except ImportError as error:
        raise ImportError("Reading BOUT dumps requires netCDF4 (pip install netCDF4)") from error

    with Dataset(source) as archive:
        def scalar(name, default):
            return int(np.asarray(archive[name][:]).item()) if name in archive.variables else default

        squashed = "squashoutput_version" in archive.ncattrs()
        if not squashed and (scalar("NXPE", 1) != 1 or scalar("NYPE", 1) != 1):
            return _read_mpi_bout_history(source, y_index, scalar("NXPE", 1) * scalar("NYPE", 1))
        missing = [name for name in ("t_array", "n", "phi", "vort")
                   if name not in archive.variables]
        if missing:
            raise KeyError(f"{source} is missing variables: {missing}")
        times = np.asarray(np.ma.filled(archive["t_array"][:], np.nan), dtype=np.float64)
        if times.ndim != 1 or not times.size or not np.all(np.isfinite(times)):
            raise ValueError(f"{source}: t_array must be a nonempty finite 1-D array")
        mxg, myg = scalar("MXG", 2), scalar("MYG", 0)
        if min(mxg, myg, y_index) < 0:
            raise ValueError(f"{source}: guard widths and y index must be non-negative")
        if squashed:
            # squashoutput retains the original decomposition/guard metadata.
            # Infer whether outer guards were collected from the global sizes.
            for axis, interior, guards in (
                ("x", scalar("nx", -1) - 2 * mxg, mxg),
                ("y", scalar("ny", -1), myg),
            ):
                if axis not in archive.dimensions:
                    continue
                extent = len(archive.dimensions[axis])
                if extent == interior:
                    if axis == "x":
                        mxg = 0
                    else:
                        myg = 0
                elif extent != interior + 2 * guards:
                    raise ValueError(
                        f"{source}: cannot infer {axis} guards in collected HW dump "
                        f"(extent={extent}, interior={interior})"
                    )
        planes = []
        for name in ("n", "phi", "vort"):
            variable = archive[name]
            dims = tuple(variable.dimensions)
            if dims not in (("t", "x", "y", "z"), ("t", "x", "z")):
                raise ValueError(f"{source}: {name} has unsupported dimensions {dims}")
            if variable.shape[0] != times.size or variable.shape[1] <= 2 * mxg:
                raise ValueError(f"{source}: invalid time or x extent for {name}")
            x_slice = slice(mxg, -mxg if mxg else None)
            if "y" in dims:
                if y_index >= variable.shape[2] - 2 * myg:
                    raise IndexError(f"{source}: y index {y_index} outside interior of {name}")
                values = variable[:, x_slice, myg + y_index, :]
            else:
                if y_index != 0:
                    raise IndexError(f"{source}: {name} has no y-plane {y_index}")
                values = variable[:, x_slice, :]
            values = np.asarray(np.ma.filled(values, np.nan), dtype=np.float64)
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{source}: {name} contains missing or non-finite values")
            if planes and values.shape != planes[0].shape:
                raise ValueError(f"{source}: inconsistent field shapes")
            planes.append(values)
    order = np.argsort(times, kind="stable")
    return times[order], np.arange(times.size)[order], np.stack(planes, axis=1)[order]


def read_history \
(
    directory: Path, batch: int, input_format: str = "auto", bout_y_index: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read NPZ snapshots or a BOUT dump, returning time, step, and fields."""
    if input_format not in ("auto", "npz", "netcdf"):
        raise ValueError(f"Unknown input format: {input_format}")
    if directory.is_file() and input_format != "npz":
        if batch != 0:
            raise IndexError("BOUT NetCDF fields are unbatched; use --bout-y-index for y planes")
        return read_bout_history(directory, bout_y_index)
    if not directory.is_dir():
        raise FileNotFoundError(f"Field directory does not exist: {directory}")
    nested_fields = directory / "fields"
    fields_directory = nested_fields if nested_fields.is_dir() else directory
    files = sorted(fields_directory.glob("fields_*.npz"))
    if input_format == "netcdf" or (input_format == "auto" and not files):
        for source in (
            directory / "BOUT.dmp.nc", directory / "BOUT.dmp.0.nc",
            directory / "data" / "BOUT.dmp.nc", directory / "data" / "BOUT.dmp.0.nc",
        ):
            if source.is_file():
                if batch != 0:
                    raise IndexError("BOUT NetCDF fields are unbatched; use --bout-y-index for y planes")
                return read_bout_history(source, bout_y_index)
        raise FileNotFoundError(f"No BOUT.dmp.nc or BOUT.dmp.0.nc found in {directory} or its data directory")
    if not files:
        raise FileNotFoundError \
        (
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
            fields = np.stack \
            (
                [_select_field(archive[name], batch, source) for name in FIELD_NAMES]
            )
            simtime = float(np.asarray(archive.get("simtime", np.nan)).item())
            step = int(np.asarray(archive.get("pStep", -1)).item())
        if not np.isfinite(simtime):
            raise ValueError(f"{source}: missing a finite simtime")
        if expected_shape is None:
            expected_shape = fields.shape
        elif fields.shape != expected_shape:
            raise ValueError \
            (
                f"{source}: field shape {fields.shape} differs from {expected_shape}"
            )
        records.append((simtime, step, fields))

    records.sort(key=lambda record: (record[0], record[1]))
    return \
    (
        np.asarray([record[0] for record in records]),
        np.asarray([record[1] for record in records]),
        np.stack([record[2] for record in records]),
    )


def periodic_derivative \
(
    values: np.ndarray, spacing: float, axis: int
) -> np.ndarray:
    """Return a fourth-order centered derivative on a periodic grid."""
    if values.shape[axis] < 5:
        raise ValueError("Fourth-order derivative requires at least five points")
    return \
    (
        -np.roll(values, -2, axis=axis)
        + 8.0 * np.roll(values, -1, axis=axis)
        - 8.0 * np.roll(values, 1, axis=axis)
        + np.roll(values, 2, axis=axis)
    ) / (12.0 * spacing)


def hw_energy_components \
(
    fields: np.ndarray, lx: float, ly: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return HW energy terms using the StylES ``L**2 * sum`` convention."""
    if min(fields.shape[-2:]) < 5:
        raise ValueError("HW energy requires at least five points per direction")
    density, potential = fields[:, 0], fields[:, 1]
    dx, dy = lx / potential.shape[-2], ly / potential.shape[-1]
    dphi_dx = periodic_derivative(potential, dx, -2)
    dphi_dy = periodic_derivative(potential, dy, -1)
    domain_factor = lx * ly
    return \
    (
        0.5 * domain_factor * np.sum(density**2, axis=(-2, -1)),
        0.5 * domain_factor * np.sum(dphi_dx**2, axis=(-2, -1)),
        0.5 * domain_factor * np.sum(dphi_dy**2, axis=(-2, -1)),
    )


def hw_particle_fluxes \
(
    fields: np.ndarray, lx: float, ly: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return radial and poloidal turbulent flux profiles with shape (t, nx).

    Fields have shape (t, channel, x, y), with U=n and V=phi. Define
    n_tilde = n - <n>_y, then Gamma_x = -<n_tilde * dphi/dy>_y and
    Gamma_y = <n_tilde * dphi/dx>_y. Both averages are over poloidal y;
    no radial or time averaging is applied here. A zonal velocity does not
    contribute because <n_tilde>_y = 0. Derivatives use the same fourth-order
    periodic stencil as the energy diagnostics.
    """
    if not np.isfinite(lx) or not np.isfinite(ly) or lx <= 0 or ly <= 0:
        raise ValueError("lx and ly must be finite and positive")
    density, potential = fields[:, 0], fields[:, 1]
    density_fluctuation = density - np.mean(density, axis=-1, keepdims=True)
    dx, dy = lx / potential.shape[-2], ly / potential.shape[-1]
    radial_flux = -np.mean(
        density_fluctuation * periodic_derivative(potential, dy, -1), axis=-1
    )
    poloidal_flux = np.mean(
        density_fluctuation * periodic_derivative(potential, dx, -2), axis=-1
    )
    return radial_flux, poloidal_flux


def hw_energy(fields: np.ndarray, lx: float, ly: float) -> np.ndarray:
    """Compute the StylES ``E=0.5*L**2*sum(...)`` energy convention."""
    return np.sum(hw_energy_components(fields, lx, ly), axis=0)


def hw_enstrophy(fields: np.ndarray, lx: float, ly: float) -> np.ndarray:
    """Compute StylES enstrophy: 0.5 * sum((n - vort)**2) * dx * dy."""
    density, vorticity = fields[:, 0], fields[:, 2]
    dx, dy = lx / density.shape[-2], ly / density.shape[-1]
    return 0.5 * np.sum((density - vorticity)**2, axis=(-2, -1)) * dx * dy


def order_datasets_with_reference_first(datasets, reference_hints=None):
    """Put the DNS dataset first, falling back to the original first dataset."""
    if reference_hints is None:
        reference_hints = (label for label, _, _ in datasets)
    reference_index = next \
    (
        (
            index for index, hint in enumerate(reference_hints)
            if "dns" in str(hint).casefold()
        ),
        0,
    )
    return \
    (
        datasets[reference_index],
        *datasets[:reference_index],
        *datasets[reference_index + 1:],
    )


def truncate_to_common_time(datasets):
    """Restrict histories to the time interval covered by every dataset."""
    final_time = min(float(times[-1]) for _, times, _ in datasets)
    truncated = []
    for label, times, fields in datasets:
        keep = times <= final_time
        if not np.any(keep):
            raise ValueError \
            (
                f"Dataset {label} has no snapshots at or before the common "
                f"final time {final_time:g}"
            )
        truncated.append((label, times[keep], fields[keep]))
    return tuple(truncated), final_time


def save_centerpoint_history(datasets, output_path: Path, dpi: int) -> Path:
    """Compare each field's value at the domain center versus time."""
    figure, axes = plt.subplots \
    (
        len(FIELD_NAMES), 1, figsize=(9, 10), sharex=True, squeeze=False
    )
    styles = \
    (
        ("black", "-"), ("tab:blue", "-"), ("tab:orange", "--"),
        ("tab:green", "-.")
    )
    for channel, name in enumerate(FIELD_NAMES):
        axis = axes[channel, 0]
        for (label, times, fields), (color, linestyle) in zip(datasets, styles):
            center_x = fields.shape[-2] // 2
            center_y = fields.shape[-1] // 2
            axis.plot \
            (
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


def save_field_extrema_history(datasets, output_path: Path, dpi: int) -> Path:
    """Compare each field's spatial minimum and maximum versus time."""
    figure, axes = plt.subplots \
    (
        len(FIELD_NAMES), 1, figsize=(9, 10), sharex=True, squeeze=False
    )
    colors = ("black", "tab:blue", "tab:orange", "tab:green")
    for channel, name in enumerate(FIELD_NAMES):
        axis = axes[channel, 0]
        for (label, times, fields), color in zip(datasets, colors):
            spatial_minimum = np.min(fields[:, channel], axis=(-2, -1))
            spatial_maximum = np.max(fields[:, channel], axis=(-2, -1))
            axis.plot \
            (
                times, spatial_minimum, color=color, linestyle="--",
                linewidth=1.6, label=f"{label} min",
            )
            axis.plot \
            (
                times, spatial_maximum, color=color, linestyle="-",
                linewidth=1.6, label=f"{label} max",
            )
        field_label = HW_FIELD_LABELS.get(name, name)
        axis.set_title(f"{field_label} spatial extrema")
        axis.set_ylabel(field_label)
        axis.grid(True, alpha=0.3)
    axes[0, 0].legend(ncol=2)
    axes[-1, 0].set_xlabel("simulation time")
    figure.suptitle("Field minima and maxima versus time")
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_field_pdf_comparison \
(
    datasets, output_dir: Path, dpi: int
) -> tuple[Path, ...]:
    """Compare dataset spatial PDFs at the initial and final times."""
    final_time = min(float(times[-1]) for _, times, _ in datasets)
    target_times = np.array([0.0, final_time])
    styles = \
    (
        ("black", "-"), ("tab:blue", "-"), ("tab:orange", "--"),
        ("tab:green", "-.")
    )
    selected_indices = \
    [
        [int(np.argmin(np.abs(times - target))) for target in target_times]
        for _, times, _ in datasets
    ]
    output_paths = []
    for channel, name in enumerate(FIELD_NAMES):
        selected_values = \
        [
            fields[indices, channel]
            for (_, _, fields), indices in zip(datasets, selected_indices)
        ]
        minimum = min \
        (
            float(values.min()) for values in selected_values
        )
        maximum = max \
        (
            float(values.max()) for values in selected_values
        )
        if minimum == maximum:
            padding = max(abs(minimum) * 0.01, 0.5)
            minimum -= padding
            maximum += padding
        bins = np.linspace(minimum, maximum, 65)
        figure, axes = plt.subplots \
        (
            1, 2, figsize=(12, 5), sharex=True, sharey=True,
        )
        field_label = HW_FIELD_LABELS.get(name, name)
        maximum_density = 0.0
        for time_index, (axis, target_time) in enumerate \
        (
            zip(axes.ravel(), target_times)
        ):
            for dataset, indices, values, (color, linestyle) in zip \
            (
                datasets, selected_indices, selected_values, styles
            ):
                label, times, _ = dataset
                pdf, _ = np.histogram \
                (
                    values[time_index].ravel(), bins=bins, density=True
                )
                maximum_density = max(maximum_density, float(pdf.max()))
                axis.stairs \
                (
                    pdf, bins, color=color, linestyle=linestyle,
                    linewidth=1.6,
                    label=f"{label} (t={times[indices[time_index]]:g})",
                )
            stage = "Initial" if time_index == 0 else "Final"
            axis.set_title(f"{stage}: target t={target_time:g}")
            axis.set_xlabel(field_label)
            axis.set_ylabel("Probability density")
            axis.grid(True, alpha=0.3)
            axis.legend(fontsize="small")
        # Set the shared scale after both panels so a taller final peak is visible.
        axes[0].set_ylim(0.0, 1.1 * maximum_density if maximum_density else 1.0)
        figure.suptitle(f"{field_label} initial and final spatial PDFs")
        figure.tight_layout()
        output_path = output_dir / f"{name}_pdf_vs_time.png"
        figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        output_paths.append(output_path)
    return tuple(output_paths)


def save_energy_comparison \
(
    datasets, output_path: Path, lx: float, ly: float, dpi: int
) -> Path:
    """Plot overlaid HW energy histories."""
    figure, axis = plt.subplots(figsize=(7.4, 5.2))
    styles = \
    (
        ("black", "-"), ("tab:blue", "-"), ("tab:orange", "--"),
        ("tab:green", "-.")
    )
    for (label, times, fields), (color, linestyle) in zip(datasets, styles):
        axis.plot \
        (
            times, hw_energy(fields, lx, ly), color=color,
            linestyle=linestyle, linewidth=1.6, label=label,
        )
    axis.set_xlabel("simulation time")
    axis.set_ylabel(r"$E=\frac{1}{2}L_xL_y\sum(n^2+|\nabla\phi|^2)$")
    axis.set_ylim(bottom=0.0)
    axis.set_title("Hasegawa–Wakatani energy versus time")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_enstrophy_comparison \
(
    datasets, output_path: Path, lx: float, ly: float, dpi: int
) -> Path:
    """Plot overlaid HW enstrophy histories using the StylES convention."""
    figure, axis = plt.subplots(figsize=(7.4, 5.2))
    styles = \
    (
        ("black", "-"), ("tab:blue", "-"), ("tab:orange", "--"),
        ("tab:green", "-.")
    )
    for (label, times, fields), (color, linestyle) in zip(datasets, styles):
        axis.plot \
        (
            times, hw_enstrophy(fields, lx, ly), color=color,
            linestyle=linestyle, linewidth=1.6, label=label,
        )
    axis.set_xlabel("simulation time")
    axis.set_ylabel(r"$Z=\frac{1}{2}\sum(n-\omega)^2\,\Delta x\,\Delta y$")
    axis.set_ylim(bottom=0.0)
    axis.set_title("Hasegawa–Wakatani enstrophy versus time")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_particle_flux_comparison \
(
    datasets, output_path: Path, lx: float, ly: float, dpi: int
) -> Path:
    """Compare radial averages of the poloidally averaged particle fluxes."""
    figure, axes = plt.subplots(2, 1, figsize=(8.2, 7.5), sharex=True)
    styles = (
        ("black", "-"), ("tab:blue", "-"), ("tab:orange", "--"),
        ("tab:green", "-.")
    )
    for (label, times, fields), (color, linestyle) in zip(datasets, styles):
        for axis, profile in zip(axes, hw_particle_fluxes(fields, lx, ly)):
            axis.plot(
                times, np.mean(profile, axis=-1), color=color,
                linestyle=linestyle, linewidth=1.6, label=label,
            )
    labels = (
        r"$\langle\Gamma_x\rangle_x=-\langle\widetilde{n}\,\partial_y\phi\rangle_{x,y}$",
        r"$\langle\Gamma_y\rangle_x=\langle\widetilde{n}\,\partial_x\phi\rangle_{x,y}$",
    )
    for axis, title, ylabel in zip(
        axes, ("Radial particle flux", "Poloidal particle flux"), labels
    ):
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.axhline(0.0, color="gray", linewidth=0.8, linestyle=":")
        axis.grid(True, alpha=0.3)
        axis.legend()
    axes[-1].set_xlabel("simulation time")
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_energy_components_comparison \
(
    datasets, output_path: Path, lx: float, ly: float, dpi: int
) -> Path:
    """Plot each additive HW energy component versus time."""
    component_labels = \
    (
        r"Density: $E_n=\frac{1}{2}\langle n^2\rangle$",
        r"Potential x-gradient: $E_{\phi,x}=\frac{1}{2}\langle(\partial_x\phi)^2\rangle$",
        r"Potential y-gradient: $E_{\phi,y}=\frac{1}{2}\langle(\partial_y\phi)^2\rangle$",
    )
    figure, axes = plt.subplots \
    (
        len(component_labels), 1, figsize=(8.2, 10), sharex=True,
        squeeze=False,
    )
    styles = \
    (
        ("black", "-"), ("tab:blue", "-"), ("tab:orange", "--"),
        ("tab:green", "-.")
    )
    for (label, times, fields), (color, linestyle) in zip(datasets, styles):
        components = hw_energy_components(fields, lx, ly)
        for axis, values in zip(axes[:, 0], components):
            axis.plot \
            (
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


def save_initial_derivative_comparison \
(
    datasets, output_path: Path, lx: float, ly: float, dpi: int
) -> Path:
    """Compare x/y derivatives of every field at the time nearest zero."""
    (first_label, first_times, first_fields), \
    (
        second_label, second_times, second_fields
    ) = datasets
    if first_fields.shape[1:] != second_fields.shape[1:]:
        raise ValueError \
        (
            "Derivative comparison requires matching field grids; received "
            f"{first_fields.shape[2:]} and {second_fields.shape[2:]}"
        )

    first_index = int(np.argmin(np.abs(first_times)))
    second_index = int(np.argmin(np.abs(second_times)))
    nx, ny = first_fields.shape[-2:]
    spacings = (lx / nx, ly / ny)
    directions = (("x", -2), ("y", -1))
    figure, axes = plt.subplots \
    (
        2 * len(FIELD_NAMES), 3, figsize=(14, 24), squeeze=False
    )
    for channel, name in enumerate(FIELD_NAMES):
        field_label = HW_FIELD_LABELS.get(name, name)
        for direction_index, ((direction, axis), spacing) in enumerate \
        (
            zip(directions, spacings)
        ):
            row = 2 * channel + direction_index
            first_values = periodic_derivative \
            (
                first_fields[first_index, channel], spacing, axis
            )
            second_values = periodic_derivative \
            (
                second_fields[second_index, channel], spacing, axis
            )
            error = np.abs(first_values - second_values)
            limit = max \
            (
                float(np.max(np.abs(first_values))),
                float(np.max(np.abs(second_values))),
            ) or 1.0
            error_limit = float(np.max(error)) or 1.0
            derivative_label = rf"$\partial_{direction}$ {field_label}"
            panels = \
            (
                (first_values, first_label, "RdBu_r", -limit, limit),
                (second_values, second_label, "RdBu_r", -limit, limit),
                (error, "absolute error", "magma", 0.0, error_limit),
            )
            for column, (values, label, cmap, vmin, vmax) in enumerate(panels):
                plot_axis = axes[row, column]
                image = plot_axis.imshow \
                (
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
    figure.suptitle \
    (
        f"Initial field derivatives: {first_label} t={first_times[first_index]:g}, "
        f"{second_label} t={second_times[second_index]:g}"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_initial_vorticity_comparison \
(
    datasets, output_path: Path, lx: float, ly: float, dpi: int
) -> Path:
    """Compare reference and two model vorticity fields nearest to t=0."""
    if len(datasets) < 3:
        raise ValueError \
        (
            "Vorticity comparison requires a reference and two model datasets."
        )
    selected = datasets[:3]
    spatial_shapes = {fields.shape[-2:] for _, _, fields in selected}
    if len(spatial_shapes) != 1:
        raise ValueError \
        (
            "Vorticity comparison requires matching field grids; received "
            f"{sorted(spatial_shapes)}"
        )

    snapshots = []
    selected_times = []
    vorticity_channel = FIELD_NAMES.index("P")
    for label, times, fields in selected:
        index = int(np.argmin(np.abs(times)))
        snapshots.append(fields[index, vorticity_channel])
        selected_times.append(float(times[index]))

    reference = snapshots[0]
    differences = \
    (
        reference - snapshots[1],
        reference - snapshots[2],
    )
    field_limit = max(float(np.max(np.abs(values))) for values in snapshots)
    difference_limit = max \
    (
        float(np.max(np.abs(values))) for values in differences
    )
    field_limit = field_limit or 1.0
    difference_limit = difference_limit or 1.0
    panels = (*snapshots, *differences)
    titles = \
    (
        selected[0][0],
        selected[1][0],
        selected[2][0],
        f"{selected[0][0]} − {selected[1][0]}",
        f"{selected[0][0]} − {selected[2][0]}",
    )

    figure, axes = plt.subplots \
    (
        1, 5, figsize=(24, 5), constrained_layout=True, squeeze=False
    )
    for column, (axis, values, title) in enumerate \
    (
        zip(axes[0], panels, titles)
    ):
        limit = field_limit if column < 3 else difference_limit
        image = axis.imshow \
        (
            values.T,
            origin="lower",
            extent=(0.0, lx, 0.0, ly),
            aspect="equal",
            interpolation="nearest",
            cmap="RdBu_r",
            vmin=-limit,
            vmax=limit,
        )
        axis.set_title(title)
        axis.set_xlabel("x")
        if column == 0:
            axis.set_ylabel("y")
        figure.colorbar(image, ax=axis, shrink=0.78)
    times_text = ", ".join \
    (
        f"{label}: t={time:g}"
        for (label, _, _), time in zip(selected, selected_times)
    )
    figure.suptitle(f"Vorticity comparison nearest t=0 ({times_text})")
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)
    return output_path


def save_field_comparison_animation \
(
    datasets, output_dir: Path, dpi: int, fps: float, overall_limits: bool = False
) -> Path:
    """Animate both 2-D solutions and their absolute error at nearest times."""
    (first_label, first_times, first_fields), \
    (
        second_label, second_times, second_fields
    ) = datasets
    first_times = first_times[:MAX_ANIMATION_FRAMES]
    first_fields = first_fields[:MAX_ANIMATION_FRAMES]
    if first_fields.shape[1:] != second_fields.shape[1:]:
        raise ValueError \
        (
            "2-D error animation requires matching field grids; received "
            f"{first_fields.shape[2:]} and {second_fields.shape[2:]}"
        )

    nearest = np.asarray \
    (
        [np.argmin(np.abs(second_times - time)) for time in first_times]
    )
    matched_second = second_fields[nearest]
    errors = np.abs(first_fields - matched_second)
    if overall_limits:
        limit_first, limit_second, limit_errors = first_fields, matched_second, errors
    else:
        first_t0 = int(np.argmin(np.abs(first_times)))
        limit_first = first_fields[first_t0:first_t0 + 1]
        limit_second = matched_second[first_t0:first_t0 + 1]
        limit_errors = errors[first_t0:first_t0 + 1]
    field_limits = \
    [
        max \
        (
            float(np.max(np.abs(limit_first[:, channel]))),
            float(np.max(np.abs(limit_second[:, channel])))
        )
        for channel in range(len(FIELD_NAMES))
    ]
    error_limits = \
    [
        float(np.max(limit_errors[:, channel]))
        for channel in range(len(FIELD_NAMES))
    ]

    frames_dir = output_dir / "field_comparison_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []
    for frame_index, (first_time, second_index) in enumerate \
    (
        zip(first_times, nearest)
    ):
        figure, axes = plt.subplots \
        (
            len(FIELD_NAMES), 3, figsize=(14, 13), squeeze=False
        )
        for channel, name in enumerate(FIELD_NAMES):
            limit = field_limits[channel] or 1.0
            error_limit = error_limits[channel] or 1.0
            panels = \
            (
                (first_fields[frame_index, channel], first_label, "RdBu_r", -limit, limit),
                (second_fields[second_index, channel], second_label, "RdBu_r", -limit, limit),
                (errors[frame_index, channel], "absolute error", "magma", 0.0, error_limit),
            )
            for column, (values, label, cmap, vmin, vmax) in enumerate(panels):
                axis = axes[channel, column]
                image = axis.imshow \
                (
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
        figure.suptitle \
        (
            f"2-D field comparison: {first_label} t={first_time:g}, "
            f"{second_label} t={second_times[second_index]:g}"
        )
        figure.tight_layout()
        frame_path = frames_dir / f"field_comparison_{frame_index:07d}.png"
        figure.savefig(frame_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        frame_paths.append(frame_path)

    return save_png_animation \
    (
        frame_paths, output_dir / "field_comparison.gif", fps=fps
    )


def save_three_dataset_field_animations \
(
    datasets, output_dir: Path, lx: float, ly: float, dpi: int, fps: float,
    overall_limits: bool = False,
) -> tuple[Path, ...]:
    """Animate each field for a reference, two models, and their differences."""
    if len(datasets) < 3:
        raise ValueError \
        (
            "Three-dataset field animations require a reference and two models."
        )
    selected = datasets[:3]
    spatial_shapes = {fields.shape[-2:] for _, _, fields in selected}
    if len(spatial_shapes) != 1:
        raise ValueError \
        (
            "Three-dataset animations require matching field grids; received "
            f"{sorted(spatial_shapes)}"
        )
    reference_label, reference_times, reference_fields = selected[0]
    reference_times = reference_times[:MAX_ANIMATION_FRAMES]
    reference_fields = reference_fields[:MAX_ANIMATION_FRAMES]
    matched = []
    matched_times = []
    for _, times, fields in selected[1:]:
        nearest = np.asarray \
        (
            [int(np.argmin(np.abs(times - time))) for time in reference_times]
        )
        matched.append(fields[nearest])
        matched_times.append(times[nearest])

    output_paths = []
    for channel, field_name in enumerate(FIELD_NAMES):
        model_one = matched[0][:, channel]
        model_two = matched[1][:, channel]
        reference = reference_fields[:, channel]
        differences = (reference - model_one, reference - model_two)
        limit_slice = slice(None) if overall_limits else slice(0, 1)
        field_limit = max \
        (
            float(np.max(np.abs(values[limit_slice])))
            for values in (reference, model_one, model_two)
        ) or 1.0
        difference_limit = max \
        (
            float(np.max(np.abs(values[limit_slice])))
            for values in differences
        ) or 1.0
        frames_directory = output_dir / f"{field_name}_comparison_frames"
        frames_directory.mkdir(parents=True, exist_ok=True)
        frame_paths = []
        for frame_index, reference_time in enumerate(reference_times):
            panels = \
            (
                reference[frame_index],
                model_one[frame_index],
                model_two[frame_index],
                differences[0][frame_index],
                differences[1][frame_index],
            )
            titles = \
            (
                reference_label,
                selected[1][0],
                selected[2][0],
                f"{reference_label} − {selected[1][0]}",
                f"{reference_label} − {selected[2][0]}",
            )
            figure, axes = plt.subplots \
            (
                1, 5, figsize=(24, 5), constrained_layout=True,
                squeeze=False,
            )
            for column, (axis, values, title) in enumerate \
            (
                zip(axes[0], panels, titles)
            ):
                limit = field_limit if column < 3 else difference_limit
                image = axis.imshow \
                (
                    values.T,
                    origin="lower",
                    extent=(0.0, lx, 0.0, ly),
                    aspect="equal",
                    interpolation="nearest",
                    cmap="RdBu_r",
                    vmin=-limit,
                    vmax=limit,
                )
                axis.set_title(title)
                axis.set_xlabel("x")
                if column == 0:
                    axis.set_ylabel("y")
                figure.colorbar(image, ax=axis, shrink=0.78)
            figure.suptitle \
            (
                f"{HW_FIELD_LABELS.get(field_name, field_name)}: "
                f"{reference_label} t={reference_time:g}; "
                f"{selected[1][0]} t={matched_times[0][frame_index]:g}; "
                f"{selected[2][0]} t={matched_times[1][frame_index]:g}"
            )
            frame_path = \
            (
                frames_directory / f"{field_name}_{frame_index:07d}.png"
            )
            figure.savefig(frame_path, dpi=dpi, bbox_inches="tight")
            plt.close(figure)
            frame_paths.append(frame_path)
        output_paths.append \
        (
            save_png_animation \
            (
                frame_paths,
                output_dir / f"{field_name}_comparison.gif",
                fps=fps,
            )
        )
    return tuple(output_paths)


def save_energy_spectrum_comparison_animation \
(
    datasets, output_dir: Path, lx: float, ly: float, dpi: int, fps: float,
    overall_limits: bool = False, smooth_spectrum: bool = False,
) -> Path:
    """Animate both datasets with black initial and blue current spectra."""
    (first_label, first_times, first_fields), \
    (
        second_label, second_times, second_fields
    ) = datasets
    first_times = first_times[:MAX_ANIMATION_FRAMES]
    first_fields = first_fields[:MAX_ANIMATION_FRAMES]
    nearest = np.asarray \
    (
        [np.argmin(np.abs(second_times - time)) for time in first_times]
    )
    first_spectra = \
    [
        _hw_energy_spectrum(fields, lx, ly, smooth_spectrum)
        for fields in first_fields
    ]
    second_spectra = \
    [
        _hw_energy_spectrum(second_fields[index], lx, ly, smooth_spectrum)
        for index in nearest
    ]

    limit_indices = range(len(first_times)) if overall_limits else \
    (
        int(np.argmin(np.abs(first_times))),
    )
    positive_energy = []
    initial_spectra = (
        _hw_energy_spectrum(first_fields[0], lx, ly, smooth_spectrum),
        _hw_energy_spectrum(second_fields[0], lx, ly, smooth_spectrum),
    )
    for _, energy in initial_spectra:
        positive_energy.extend(energy[energy > 0.0])
    for index in limit_indices:
        for _, energy in (first_spectra[index], second_spectra[index]):
            positive_energy.extend(energy[energy > 0.0])
    if positive_energy:
        ymax = float(np.max(positive_energy))
        if ymax <= SPECTRUM_Y_MIN:
            ymax = 10.0 * SPECTRUM_Y_MIN
    else:
        ymax = 10.0 * SPECTRUM_Y_MIN

    frames_dir = output_dir / "energy_spectrum_comparison_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []
    for frame_index, (first_time, second_index) in enumerate \
    (
        zip(first_times, nearest)
    ):
        figure, axis = plt.subplots(figsize=(8.2, 5.8))
        first_k, first_energy = first_spectra[frame_index]
        second_k, second_energy = second_spectra[frame_index]
        for (initial_k, initial_energy), label, initial_time, style in zip(
            initial_spectra, (first_label, second_label),
            (first_times[0], second_times[0]), ("-", "--"),
        ):
            axis.loglog(
                initial_k[1:], initial_energy[1:], color="black",
                linestyle=style, linewidth=1.8,
                label=f"{label} initial (t={initial_time:g})",
            )
        axis.loglog \
        (
            second_k[1:], second_energy[1:], color="blue", linestyle="--",
            linewidth=1.8, label=f"{second_label} current (t={second_times[second_index]:g})"
        )
        axis.loglog \
        (
            first_k[1:], first_energy[1:], color="blue",
            linewidth=1.8, label=f"{first_label} current (t={first_time:g})"
        )
        axis.set_ylim(SPECTRUM_Y_MIN, ymax)
        axis.set_xlabel(r"radial wavenumber $k$")
        axis.set_ylabel(r"HW energy spectrum $E(k)$")
        axis.set_title \
        (
            f"HW energy spectrum: {first_label} t={first_time:g}, "
            f"{second_label} t={second_times[second_index]:g}"
        )
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()
        figure.tight_layout()
        frame_path = frames_dir / f"energy_spectrum_{frame_index:07d}.png"
        figure.savefig(frame_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        frame_paths.append(frame_path)

    return save_png_animation \
    (
        frame_paths, output_dir / "energy_spectrum_comparison.gif", fps=fps
    )


def _hw_energy_spectrum \
(
    fields: np.ndarray, lx: float, ly: float, smooth: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Compute an HW spectrum, optionally using StylES five-bin smoothing."""
    return isotropic_hw_energy_spectrum(fields, lx, ly, smooth=smooth)


def save_endpoint_spectrum_comparison \
(
    datasets, path: Path, lx: float, ly: float, dpi: int,
    smooth: bool = False, cutoff_n: int = 64,
) -> Path:
    """Compare HW energy spectra nearest t=0 and at each retained final time."""
    figure, axes = plt.subplots(
        1, 2, figsize=(13, 5.5), sharex=True, sharey=True,
        constrained_layout=True,
    )
    line_styles = ("-", "--", "-.", ":")
    for axis, stage in zip(axes, ("Initial", "Final")):
        selected_times = []
        for dataset_index, (label, times, fields) in enumerate(datasets):
            time_index = int(np.argmin(np.abs(times))) if stage == "Initial" else -1
            selected_times.append(f"{label}: t={times[time_index]:g}")
            k, energy = _hw_energy_spectrum(fields[time_index], lx, ly, smooth)
            positive = (k > 0) & (energy > 0)
            axis.loglog(
                k[positive], energy[positive],
                line_styles[dataset_index % len(line_styles)],
                linewidth=2, label=label,
            )
        if cutoff_n > 0:
            cutoff = min(np.pi / (lx / cutoff_n), np.pi / (ly / cutoff_n))
            axis.axvline(
                cutoff, color="black", linestyle=":", linewidth=1.4,
                label=f"{cutoff_n}-grid Nyquist",
            )
        axis.set_title(f"{stage} HW energy spectrum\n" + "; ".join(selected_times))
        axis.set_xlabel(r"$k$")
        axis.set_ylabel(r"Shell energy $E(k)$")
        axis.set_ylim(SPECTRUM_Y_MIN, SPECTRUM_Y_MAX)
        axis.grid(True, which="both", alpha=0.25)
        axis.legend()
    figure.savefig(path, dpi=dpi)
    plt.close(figure)
    return path


def save_field_animation \
(
    dataset, output_dir: Path, dpi: int, fps: float,
    overall_limits: bool = False,
) -> Path:
    """Animate the fields from one dataset without comparison panels."""
    label, times, fields = dataset
    times = times[:MAX_ANIMATION_FRAMES]
    fields = fields[:MAX_ANIMATION_FRAMES]
    limit_fields = fields if overall_limits else fields \
    [
        int(np.argmin(np.abs(times))):int(np.argmin(np.abs(times))) + 1
    ]
    field_limits = \
    [
        float(np.max(np.abs(limit_fields[:, channel]))) or 1.0
        for channel in range(len(FIELD_NAMES))
    ]

    frames_dir = output_dir / "field_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []
    for frame_index, time in enumerate(times):
        figure, axes = plt.subplots(1, len(FIELD_NAMES), figsize=(14, 4.6))
        for channel, name in enumerate(FIELD_NAMES):
            limit = field_limits[channel]
            image = axes[channel].imshow \
            (
                fields[frame_index, channel].T,
                origin="lower",
                extent=(0.0, 1.0, 0.0, 1.0),
                aspect="equal",
                interpolation="nearest",
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
            )
            axes[channel].set_title(HW_FIELD_LABELS.get(name, name))
            axes[channel].set_xlabel("x / Lx")
            axes[channel].set_ylabel("y / Ly")
            figure.colorbar(image, ax=axes[channel], shrink=0.78)
        figure.suptitle(f"{label}: t={time:g}")
        figure.tight_layout()
        frame_path = frames_dir / f"fields_{frame_index:07d}.png"
        figure.savefig(frame_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        frame_paths.append(frame_path)

    return save_png_animation \
    (
        frame_paths, output_dir / "field_animation.gif", fps=fps
    )


def save_energy_spectrum_animation \
(
    dataset, output_dir: Path, lx: float, ly: float, dpi: int, fps: float,
    overall_limits: bool = False, smooth_spectrum: bool = False,
) -> Path:
    """Animate current HW spectra in blue against the fixed initial in black."""
    label, times, fields = dataset
    times = times[:MAX_ANIMATION_FRAMES]
    fields = fields[:MAX_ANIMATION_FRAMES]
    spectra = \
    [
        _hw_energy_spectrum(frame, lx, ly, smooth_spectrum) for frame in fields
    ]
    initial_k, initial_energy = spectra[0]
    limit_indices = range(len(times)) if overall_limits else \
    (
        int(np.argmin(np.abs(times))),
    )
    positive_energy = np.concatenate \
    (
         \
        [
            spectra[index][1][spectra[index][1] > 0.0]
            for index in sorted(set(limit_indices) | {0})
        ]
    )
    ymax = \
    (
        float(np.max(positive_energy)) if positive_energy.size
        else 10.0 * SPECTRUM_Y_MIN
    )
    ymax = max(ymax, 10.0 * SPECTRUM_Y_MIN)

    frames_dir = output_dir / "energy_spectrum_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_paths = []
    for frame_index, (time, (wavenumber, energy)) in enumerate \
    (
        zip(times, spectra)
    ):
        figure, axis = plt.subplots(figsize=(8.2, 5.8))
        axis.loglog \
        (
            initial_k[1:], initial_energy[1:], color="black",
            linewidth=1.8, label=f"Initial (t={times[0]:g})"
        )
        axis.loglog \
        (
            wavenumber[1:], energy[1:], color="blue",
            linewidth=1.8, label=f"Current (t={time:g})"
        )
        axis.set_ylim(SPECTRUM_Y_MIN, ymax)
        axis.set_xlabel(r"radial wavenumber $k$")
        axis.set_ylabel(r"HW energy spectrum $E(k)$")
        axis.set_title(f"HW energy spectrum: {label} t={time:g}")
        axis.grid(True, which="both", alpha=0.3)
        axis.legend()
        figure.tight_layout()
        frame_path = frames_dir / f"energy_spectrum_{frame_index:07d}.png"
        figure.savefig(frame_path, dpi=dpi, bbox_inches="tight")
        plt.close(figure)
        frame_paths.append(frame_path)

    return save_png_animation \
    (
        frame_paths, output_dir / "energy_spectrum_animation.gif", fps=fps
    )


def main() -> None:
    args = parse_args()
    if args.lx <= 0 or args.ly <= 0:
        raise ValueError("lx and ly must be positive")
    if args.batch < 0:
        raise ValueError("batch must be non-negative")
    if args.spectrum_cutoff_n < 0:
        raise ValueError("spectrum-cutoff-n must be non-negative")
    if args.animations and args.fps <= 0:
        raise ValueError("fps must be positive")

    optional_paths = (args.dataset2_fields, args.dataset3_fields)
    dataset_paths = (args.unet_fields, args.dns_fields, *optional_paths)
    requested_paths = tuple(path for path in dataset_paths if path is not None)
    input_paths = tuple \
    (
        resolve_input_directory(path, args.base_dir)
        for path in requested_paths
    )
    datasets = []
    for dataset_path in input_paths:
        times, _, fields = read_history(
            dataset_path, args.batch, args.input_format, args.bout_y_index
        )
        label_path = dataset_path.parent if dataset_path.is_file() else dataset_path
        label = label_path.resolve().name.replace("results_U-Net_", "")
        if "dns" in str(dataset_path.resolve()).casefold():
            label = "DNS"
        datasets.append((label, times, fields))
    reference_hints = tuple(path.resolve() for path in input_paths)
    datasets = order_datasets_with_reference_first \
    (
        tuple(datasets), reference_hints
    )
    datasets, common_final_time = truncate_to_common_time(datasets)
    print(f"Restricting all datasets to t <= {common_final_time:g}")
    comparison_datasets = \
    (
        (datasets[1], datasets[0])
        if len(datasets) > 1 else None
    )
    output_directory = args.output_dir.resolve()
    input_directories = tuple \
    (
        path.resolve() for path in input_paths
    )
    if any \
    (
        input_directory == output_directory
        or output_directory in input_directory.parents
        for input_directory in input_directories
    ):
        raise ValueError \
        (
            f"Refusing to delete output directory {output_directory} because "
            "it contains an input dataset"
        )
    if args.output_dir.exists():
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    centerpoint_path = save_centerpoint_history \
    (
        datasets, args.output_dir / "centerpoint_vs_time.png", args.dpi
    )
    extrema_path = save_field_extrema_history \
    (
        datasets, args.output_dir / "field_minmax_vs_time.png", args.dpi
    )
    energy_path = save_energy_comparison \
    (
        datasets, args.output_dir / "energy_vs_time.png",
        args.lx, args.ly, args.dpi,
    )
    enstrophy_path = save_enstrophy_comparison \
    (
        datasets, args.output_dir / "enstrophy_vs_time.png",
        args.lx, args.ly, args.dpi,
    )
    particle_flux_path = save_particle_flux_comparison \
    (
        datasets, args.output_dir / "particle_flux_vs_time.png",
        args.lx, args.ly, args.dpi,
    )
    energy_components_path = save_energy_components_comparison \
    (
        datasets, args.output_dir / "energy_components_vs_time.png",
        args.lx, args.ly, args.dpi,
    )
    endpoint_spectra_path = save_endpoint_spectrum_comparison \
    (
        datasets,
        args.output_dir / "hw_energy_spectra_initial_final.png",
        args.lx,
        args.ly,
        args.dpi,
        args.smooth_spectrum,
        args.spectrum_cutoff_n,
    )
    print(f"Saved {centerpoint_path}")
    print(f"Saved {extrema_path}")
    print(f"Saved {energy_path}")
    print(f"Saved {enstrophy_path}")
    print(f"Saved {particle_flux_path}")
    print(f"Saved {energy_components_path}")
    print(f"Saved {endpoint_spectra_path}")
    pdf_paths = save_field_pdf_comparison(datasets, args.output_dir, args.dpi)
    for pdf_path in pdf_paths:
        print(f"Saved {pdf_path}")
    if len(datasets) >= 3:
        vorticity_path = save_initial_vorticity_comparison \
        (
            datasets,
            args.output_dir / "vorticity_t0_comparison.png",
            args.lx,
            args.ly,
            args.dpi,
        )
        print(f"Saved {vorticity_path}")
    if comparison_datasets is not None:
        initial_derivatives_path = save_initial_derivative_comparison \
        (
            comparison_datasets, args.output_dir / "field_derivatives_t0.png",
            args.lx, args.ly, args.dpi,
        )
        print(f"Saved {initial_derivatives_path}")
    if args.animations:
        if comparison_datasets is None:
            animation_path = save_field_animation \
            (
                datasets[0], args.output_dir, args.dpi, args.fps,
                args.overall_animation_limits,
            )
            spectrum_animation_path = save_energy_spectrum_animation \
            (
                datasets[0], args.output_dir, args.lx, args.ly, args.dpi,
                args.fps, args.overall_animation_limits, args.smooth_spectrum,
            )
        else:
            animation_path = save_field_comparison_animation \
            (
                comparison_datasets, args.output_dir, args.dpi, args.fps,
                args.overall_animation_limits,
            )
            spectrum_animation_path = save_energy_spectrum_comparison_animation \
            (
                comparison_datasets, args.output_dir, args.lx, args.ly,
                args.dpi, args.fps, args.overall_animation_limits,
                args.smooth_spectrum,
            )
        print(f"Saved {animation_path}")
        print(f"Saved {spectrum_animation_path}")
        if len(datasets) >= 3:
            field_animation_paths = save_three_dataset_field_animations \
            (
                datasets,
                args.output_dir,
                args.lx,
                args.ly,
                args.dpi,
                args.fps,
                args.overall_animation_limits,
            )
            for field_animation_path in field_animation_paths:
                print(f"Saved {field_animation_path}")


if __name__ == "__main__":
    main()
