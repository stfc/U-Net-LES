#!/usr/bin/env python3
"""Compare HW particle fluxes from four BOUT_runs simulations.

Requires numpy, matplotlib and netCDF4. Example from the repository root:
    python miscellaneous/compare_hw_fluxes.py --t-min 200 --t-max 1000

BOUT X is radial; BOUT Z is poloidal in these 2D runs. Fourth-order
centered derivatives use communicated X guard cells for native dumps
and periodic derivatives in both directions for reconstructed snapshots.
Full-density fluxes are (-<n dphi/dz>, <n dphi/dx>); turbulent fluxes
replace n with n - <n>_z. All reported fluxes are spatial means.
U-Net fluxes use reconstructed results_U-Net/fields NPZ outputs; other
runs use native-grid data dumps. No smoothing, time shifting or amplitude
rescaling is applied.
"""
import argparse
import configparser
import csv
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/hw_flux_matplotlib')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from netCDF4 import Dataset

ROOT = Path(__file__).resolve().parents[1]
RUNS = ('HW_DNS512', 'HW_DNS512_LES64',
        'HW_DNS512_LES64_CENTORI', 'HW_DNS512_LES64_SMAGORINSKY')
LABELS = ('DNS 512', 'LES 64 (U-Net reconstructed)', 'LES 64 (CENTORI)',
          'LES 64 (Smagorinsky)')
COLUMNS = ('radial_full', 'poloidal_full', 'radial_turbulent', 'poloidal_turbulent')


def derivative(values, spacing, axis):
    return (np.roll(values, 2, axis) - 8 * np.roll(values, 1, axis)
            + 8 * np.roll(values, -1, axis) - np.roll(values, -2, axis)) / (12 * spacing)


def read_unet_flux(folder, t_min, t_max, stride):
    """Reduce reconstructed periodic fields, ordered by recorded simulation time."""
    config = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
    config.read_string('[bout_root]\n' + (folder / 'data' / 'BOUT.inp').read_text())
    lx, lz = (config.getfloat('mesh', key) for key in ('Lx', 'Lz'))
    if not np.isfinite([lx, lz]).all() or min(lx, lz) <= 0:
        raise ValueError(f'{folder}: invalid domain lengths')
    samples = []
    for path in (folder / 'results_U-Net' / 'fields').glob('fields_*.npz'):
        with np.load(path) as dump:
            time = float(dump['simtime'])
        if not np.isfinite(time):
            raise ValueError(f'{path}: nonfinite simulation time')
        samples.append((time, path))
    samples.sort()
    times = np.array([time for time, _ in samples])
    if np.any(np.diff(times) <= 0):
        raise ValueError(f'{folder}: times must be strictly increasing')
    selected = [(time, path) for time, path in samples if t_min <= time <= t_max][::stride]
    if not selected:
        raise ValueError(f'{folder}: no samples in requested time interval')
    rows = []
    for time, path in selected:
        with np.load(path) as dump:
            n = np.asarray(dump['U'], dtype=float)
            phi = np.asarray(dump['V'], dtype=float)
        if n.ndim != 2 or phi.shape != n.shape or min(n.shape) < 5:
            raise ValueError(f'{path}: incompatible reconstructed field shapes')
        if not np.isfinite(n).all() or not np.isfinite(phi).all():
            raise ValueError(f'{path}: nonfinite field values')
        px = derivative(phi, lx / n.shape[0], 0)
        pz = derivative(phi, lz / n.shape[1], 1)
        fluct = n - n.mean(axis=1, keepdims=True)
        rows.append((time, (-n * pz).mean(), (n * px).mean(),
                     (-fluct * pz).mean(), (fluct * px).mean()))
    return np.asarray(rows)


def read_flux(folder, t_min, t_max, stride, chunk_size):
    """Reduce rank-local chunks; guard cells are excluded from all means."""
    config = configparser.ConfigParser(inline_comment_prefixes=('#', ';'))
    config.read_string('[bout_root]\n' + (folder / 'BOUT.inp').read_text())
    lx, lz = (config.getfloat('mesh', key) for key in ('Lx', 'Lz'))
    if not np.isfinite([lx, lz]).all() or min(lx, lz) <= 0:
        raise ValueError(f'{folder}: invalid domain lengths')
    with Dataset(folder / 'BOUT.dmp.0.nc') as dump:
        times = np.asarray(dump['t_array'][:], dtype=float)
        ranks = int(dump['NXPE'][...])
        local_nx = int(dump['MXSUB'][...])
        nz = dump['phi'].shape[-1]
        if int(dump['NYPE'][...]) != 1 or dump['phi'].shape[2] != 1:
            raise ValueError('Only the X-decomposed 2D HW geometry is supported')
    if not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError(f'{folder}: times must be finite and strictly increasing')
    indices = np.flatnonzero((times >= t_min) & (times <= t_max))[::stride]
    if not indices.size:
        raise ValueError(f'{folder}: no samples in requested time interval')
    nx = ranks * local_nx
    if min(nx, nz) < 5:
        raise ValueError('Fourth-order derivatives need at least five grid points')
    result = np.zeros((len(indices), 4))
    for rank in range(ranks):
        with Dataset(folder / f'BOUT.dmp.{rank}.nc') as dump:
            guard = int(dump['MXG'][...])
            if guard < 2 or int(dump['MXSUB'][...]) != local_nx:
                raise ValueError(f'{folder}: incompatible X decomposition')
            if not np.array_equal(times, dump['t_array'][:]):
                raise ValueError(f'{folder}: rank timestamps differ')
            expected = (len(times), local_nx + 2 * guard, 1, nz)
            if dump['n'].shape != expected or dump['phi'].shape != expected:
                raise ValueError(f'{folder}: unexpected field shape on rank {rank}')
            for start in range(0, len(indices), chunk_size):
                stop = min(start + chunk_size, len(indices))
                selected = indices[start:stop]
                n = np.asarray(dump['n'][selected, :, 0, :], dtype=float)
                phi = np.asarray(dump['phi'][selected, :, 0, :], dtype=float)
                if not np.isfinite(n).all() or not np.isfinite(phi).all():
                    raise ValueError(f'{folder}: nonfinite field values')
                px = derivative(phi, lx / nx, 1)[:, guard:-guard]
                pz = derivative(phi, lz / nz, 2)[:, guard:-guard]
                n = n[:, guard:-guard]
                fluct = n - n.mean(axis=2, keepdims=True)
                for col, flux in enumerate((-n * pz, n * px, -fluct * pz, fluct * px)):
                    result[start:stop, col] += flux.sum(axis=(1, 2)) / (nx * nz)
    return np.column_stack((times[indices], result))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', nargs='+', choices=RUNS, default=list(RUNS),
                        help='Runs to compare (default: all four)')
    parser.add_argument('--runs-dir', type=Path, default=ROOT / 'BOUT_runs')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'BOUT_runs' / 'hw_flux_comparison')
    parser.add_argument('--t-min', type=float, default=-np.inf)
    parser.add_argument('--t-max', type=float, default=np.inf)
    parser.add_argument('--stride', type=int, default=1, help='Read every Nth selected output')
    parser.add_argument('--chunk-size', type=int, default=8)
    args = parser.parse_args()
    if args.stride < 1 or args.chunk_size < 1 or args.t_min > args.t_max:
        parser.error('stride/chunk-size must be positive and t-min <= t-max')
    for run in args.runs:
        if run == 'HW_DNS512_LES64':
            path = args.runs_dir / run / 'results_U-Net' / 'fields'
            if not any(path.glob('fields_*.npz')):
                parser.error(f'Missing reconstructed fields: {path}. Use --runs to select available runs.')
            continue
        path = args.runs_dir / run / 'data' / 'BOUT.dmp.0.nc'
        if not path.is_file():
            parser.error(f'Missing field dump: {path}. Use --runs to select available runs.')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    datasets = []
    for run in args.runs:
        print(f'Reading {run} ...', flush=True)
        if run == 'HW_DNS512_LES64':
            data = read_unet_flux(args.runs_dir / run, args.t_min, args.t_max, args.stride)
        else:
            data = read_flux(args.runs_dir / run / 'data', args.t_min, args.t_max,
                             args.stride, args.chunk_size)
        datasets.append(data)
        np.savetxt(args.output_dir / f'{run}_flux.csv', data, delimiter=',',
                   header='time,' + ','.join(COLUMNS), comments='')
    # Compare statistics over the common recorded interval, using time weights.
    lower = max(data[0, 0] for data in datasets)
    upper = min(data[-1, 0] for data in datasets)
    if lower > upper:
        raise ValueError('Runs have no overlapping recorded time interval')
    with (args.output_dir / 'flux_summary.csv').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(('run', 't_start', 't_end', 'component', 'time_mean', 'time_std'))
        for run, data in zip(args.runs, datasets):
            t = np.unique(np.concatenate(([lower], data[(data[:, 0] > lower) &
                                                         (data[:, 0] < upper), 0], [upper])))
            for col, name in enumerate(COLUMNS, 1):
                values = np.interp(t, data[:, 0], data[:, col])
                mean = np.sum(0.5 * (values[1:] + values[:-1]) * np.diff(t)) / (upper - lower) if upper > lower else values[0]
                variance = np.sum(0.5 * ((values[1:] - mean)**2 + (values[:-1] - mean)**2) * np.diff(t)) / (upper - lower) if upper > lower else 0
                writer.writerow((run, lower, upper, name, mean, np.sqrt(variance)))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for column, ax in enumerate(axes.T.flat):
        for label, data in zip((LABELS[RUNS.index(run)] for run in args.runs), datasets):
            ax.plot(data[:, 0], data[:, column + 1], label=label, linewidth=1.2)
        ax.set_title(COLUMNS[column].replace('_', ' ').capitalize())
        ax.set_ylabel(r'$\Gamma_x$' if column % 2 == 0 else r'$\Gamma_{\mathrm{pol}}$')
        ax.axhline(0, color='grey', linewidth=0.5)
        ax.grid(alpha=0.2)
    for ax in axes[1]:
        ax.set_xlabel('Recorded simulation time')
    axes[0, 0].legend(fontsize=9)
    fig.suptitle('HW particle fluxes: spatial means (U-Net reconstructed fields)')
    fig.tight_layout()
    for extension in ('png', 'pdf'):
        fig.savefig(args.output_dir / f'particle_flux_comparison.{extension}', dpi=200)
    plt.close(fig)
    print(f'Saved plots, flux histories and common-interval statistics to {args.output_dir}')


if __name__ == '__main__':
    main()
