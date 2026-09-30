#!/usr/bin/env python3
"""Create native 32-task BOUT++ restart files from a PoP23 test snapshot.

Run inside a 32-task Slurm allocation. A zero-step native run supplies the
restart metadata; replace its evolved fields with periodic slices of the NPZ.
The production run then loads these files using BOUT++'s `restart` option.
"""

import argparse
import configparser
import io
from pathlib import Path
import shutil
import subprocess
import tempfile

import numpy as np
from netCDF4 import Dataset


def prepare(run_dir, source, fine_n):
    ranks, guards = 32, 2
    if fine_n % ranks or fine_n // ranks < guards:
        raise ValueError(f'Cannot split grid {fine_n} across {ranks} X ranks')
    with np.load(source) as snapshot:
        fields = {name: np.asarray(snapshot[key], dtype=np.float64)
                  for name, key in (('n', 'U'), ('vort', 'P'))}
        # Potential is reconstructed by the native DNS Poisson solver.
        potential = np.asarray(snapshot['V'], dtype=np.float64)
    for name, field in {**fields, 'phi': potential}.items():
        if field.shape != (fine_n, fine_n) or not np.isfinite(field).all():
            raise ValueError(f'{source}: invalid {name} field (shape {field.shape})')

    data = run_dir / 'data'
    if list(data.glob('BOUT.restart.*')):
        raise FileExistsError(f'Refusing to overwrite existing restart files in {data}')
    # Keep failed attempts for diagnosis while allowing preparation to be retried.
    seed = Path(tempfile.mkdtemp(prefix='restart_seed_', dir=run_dir))
    config = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=('#',))
    config.optionxform = str
    config.read_string('[root]\n' + (data / 'BOUT.inp').read_text())
    # Match pBOUT-unet.initFlow: mean-free density/potential and omega=Delp2(phi).
    # The launcher explicitly uses the second-order periodic Laplacian for LES.
    fields['n'] -= fields['n'].mean()
    potential = potential - potential.mean()
    dx = float(config['mesh']['Lx']) / fine_n
    dz = float(config['mesh']['Lz']) / fine_n
    fields['vort'] = (
        (np.roll(potential, 1, axis=0) - 2 * potential + np.roll(potential, -1, axis=0)) / dx**2
        + (np.roll(potential, 1, axis=1) - 2 * potential + np.roll(potential, -1, axis=1)) / dz**2
    )
    # BOUT++ derives NYPE from NPES/NXPE and rejects an explicit NYPE as unused.
    config['root'].pop('NYPE', None)
    config['root'].update(nout='0', NXPE=str(ranks), restart='false')
    config['solver']['nout'] = '0'
    config['unet']['dns_init'] = '-1'
    config['output']['enabled'] = 'false'
    output = io.StringIO()
    config.write(output)
    (seed / 'BOUT.inp').write_text(output.getvalue().removeprefix('[root]\n'))
    subprocess.run(['mpirun', '-np', str(ranks), str(run_dir / 'hasegawa-wakatani'),
                    '-d', str(seed)], cwd=run_dir, check=True)

    width = fine_n // ranks
    # Finish and validate every rank before copying any files to the production directory.
    for rank in range(ranks):
        path = seed / f'BOUT.restart.{rank}.nc'
        if not path.is_file():
            raise FileNotFoundError(path)
        with Dataset(path, 'r+') as restart:
            if int(restart['NXPE'][...]) != ranks or int(restart['NYPE'][...]) != 1:
                raise ValueError(f'{path}: unexpected processor layout')
            indices = np.arange(rank * width - guards, (rank + 1) * width + guards) % fine_n
            for name, field in fields.items():
                values = field[indices, None, :]
                if restart[name].shape != values.shape:
                    raise ValueError(f'{path}: {name} shape {restart[name].shape} != {values.shape}')
                restart[name][:] = values
    for rank in range(ranks):
        name = f'BOUT.restart.{rank}.nc'
        shutil.copy2(seed / name, data / name)
    print(f'Prepared {ranks} MPI restart files from {source}; elapsed time starts at zero.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run_dir', type=Path)
    parser.add_argument('source', type=Path)
    parser.add_argument('--fine-n', type=int, required=True)
    args = parser.parse_args()
    prepare(args.run_dir.resolve(), args.source.resolve(), args.fine_n)


if __name__ == '__main__':
    main()
