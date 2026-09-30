#!/usr/bin/env python3
"""Build and submit HW/mHW DNS256/512 and factor 2/4/8 LES via HW_template/run.sh.

Run from BOUT_runs: python launch_hw_runs.py [--dry-run]
DNS and LES start from the same held-out dataset field for each model/resolution.
A preparation job creates 32 MPI restart files before DNS starts (afterok).
Existing run directories are skipped. No jobs are submitted until all new builds
succeed. DNS and LES both evolve for 1000 time units by default, with the
selected field defining elapsed time zero.
All simulation jobs use 32 CPU cores, including DNS512.
"""
import argparse
import configparser
import json
import random
from pathlib import Path
import shlex
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / 'HW_template'
REPO = ROOT.parent
DATASET = REPO.parent.parent / 'data/BOUT_runs/HW_2D/Papers/PoP23'


def test_files(dataset, parameters):
    """Match train_germano_unet.split_files, including explicit split folders."""
    def files(path):
        return sorted(p for p in path.rglob('*.npz') if p.is_file())

    if all((dataset / split).is_dir() for split in ('train', 'validation', 'test')):
        return files(dataset / 'test')
    snapshots = files(dataset)
    if len(snapshots) < 3:
        raise ValueError(f'At least three NPZ snapshots are required: {dataset}')
    if parameters['split_mode'] == 'random':
        random.Random(parameters['seed']).shuffle(snapshots)
    elif parameters['split_mode'] != 'sequential':
        raise ValueError(f"Unknown split mode: {parameters['split_mode']}")
    nv = max(1, round(parameters['validation_fraction'] * len(snapshots)))
    nt = max(1, round(parameters['test_fraction'] * len(snapshots)))
    ntr = len(snapshots) - nv - nt
    if ntr < 1:
        raise ValueError('The split leaves no training snapshots.')
    return snapshots[ntr + nv:]


def select_test_field(dataset_root, model, fine):
    """Use a field held out by all three LES training runs."""
    candidates = None
    for factor in (2, 4, 8):
        parameters_path = (REPO / 'training' /
            f'training_{model}_dns{fine}_les{fine // factor}' / 'training_parameters.json')
        parameters = json.loads(parameters_path.read_text())
        held_out = test_files(dataset_root / f'{model}_N{fine}' / 'fields', parameters)
        held_out_set = set(held_out)
        candidates = held_out if candidates is None else [p for p in candidates if p in held_out_set]
    if not candidates:
        raise ValueError(f'No common test field for {model} DNS{fine}')
    return candidates[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dry-run', action='store_true', help='Print plan without writing, building or submitting')
    parser.add_argument('--prefix', default='HW', help='HW run prefix; modified runs prepend m (default: HW and mHW)')
    parser.add_argument('--dns-nout', type=int, default=1000)
    parser.add_argument('--les-nout', type=int, default=1000)
    parser.add_argument('--dataset-root', type=Path, default=DATASET, help='PoP23 dataset root')
    args = parser.parse_args()
    if not args.prefix or Path(args.prefix).name != args.prefix or args.prefix in ('.', '..'):
        parser.error('prefix must be a directory name, not a path')
    for value in (args.dns_nout, args.les_nout):
        if value is not None and value <= 0:
            parser.error('nout must be positive')

    plans = []
    for model, modified in (('HW', False), ('mHW', True)):
        prefix = f'm{args.prefix}' if modified else args.prefix
        for fine in (256, 512):
            dns_dir = ROOT / f'{prefix}_DNS{fine}'
            source = None
            for factor in (1, 2, 4, 8):
                coarse = fine // factor
                run_dir = dns_dir if factor == 1 else ROOT / f'{prefix}_DNS{fine}_LES{coarse}'
                if run_dir.exists():
                    print(f'Skipping existing case: {run_dir.name}', flush=True)
                    continue
                checkpoint = None if factor == 1 else (
                    REPO / f'training/training_{model}_dns{fine}_les{coarse}/checkpoints/best.pt'
                )
                if source is None:
                    try:
                        source = select_test_field(args.dataset_root.resolve(), model, fine)
                    except (OSError, ValueError, KeyError) as error:
                        parser.error(str(error))
                if checkpoint is not None and not checkpoint.is_file():
                    parser.error(f'Missing checkpoint: {checkpoint}')
                plans.append((model, modified, fine, factor, coarse, run_dir, checkpoint, source))
    if not plans:
        print('No new cases to launch.')
        return
    if not args.dry_run and not shutil.which('sbatch'):
        parser.error('sbatch is not available')

    for model, modified, fine, factor, coarse, run_dir, checkpoint, source in plans:
        print(f'{run_dir.name}: grid={coarse}, kappa={1.0 if modified else fine / 512:g}, modified={str(modified).lower()}, dns_init={-1 if factor == 1 else 0}, ranks={32 if factor == 1 else 1}', flush=True)
        print(f'  checkpoint={checkpoint}\n  dns_path={source}', flush=True)
        if args.dry_run:
            continue
        (run_dir / 'data').mkdir(parents=True)
        # A synthetic section preserves BOUT's top-level options with ConfigParser.
        config = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=('#',))
        config.optionxform = str
        config.read_string('[root]\n' + (TEMPLATE / 'data/BOUT.inp').read_text())
        config['mesh'].update(nx=str(coarse + 4), nz=str(coarse))
        # Set the decomposition explicitly to match the prepared restart files.
        for section in config.values():
            for key in list(section):
                if key.upper() in {'NXPE', 'NYPE', 'NZPE'}:
                    del section[key]
        config['root']['NXPE'] = str(32 if factor == 1 else 1)
        config['root']['timestep'] = '1'
        nout = args.dns_nout if factor == 1 else args.les_nout
        if nout is not None:
            config['root']['nout'] = str(nout)
        config['hw']['kappa'] = str(1.0 if modified else fine / 512)
        config['hw']['modified'] = str(modified).lower()
        config['unet'].update(dns_init=str(-1 if factor == 1 else 0),
                             device='cpu' if factor == 1 else 'cuda',
                             write_diagnostics='0' if factor == 1 else '2',
                             write_arakawa_terms='false', laplacian_order='2')
        if factor == 1:
            config['unet'].pop('checkpoint', None)
            config['unet'].pop('dns_path', None)
        else:
            config['unet'].update(checkpoint=str(checkpoint), dns_path=str(source))
        import io
        output = io.StringIO()
        config.write(output)
        (run_dir / 'data/BOUT.inp').write_text(output.getvalue().removeprefix('[root]\n'))
        (run_dir / 'initial_field.txt').write_text(str(source) + '\n')
        subprocess.run(['bash', str(TEMPLATE / 'build.sh'), str(run_dir)], check=True)
        (run_dir / 'hasegawa-wakatani').symlink_to('build_release/hasegawa-wakatani')

    def submit(run_dir, name, ranks, cpus, dependency=None, extra=(), nodes=1):
        # Save the complete submission (resources, dependency and command) locally.
        script = run_dir / ('prepare_restart.sh' if extra else 'run.sh')
        directives = {
            'job-name': name, 'output': f'{name}-%j.out', 'error': f'{name}-%j.err',
            'nodes': str(nodes),
            'ntasks': str(ranks), 'ntasks-per-node': str(ranks // nodes),
            'cpus-per-task': str(cpus),
            'gpus-per-node': str(1 if name.startswith('LES') else 0),
        }
        if dependency:
            directives['dependency'] = f'afterok:{dependency}'
        template = (TEMPLATE / 'run.sh').read_text()
        header, body = template.split('set -euo pipefail', 1)
        header = '\n'.join(line for line in header.splitlines()
                           if not any(line.startswith(f'#SBATCH --{key}=')
                                      for key in directives))
        header += '\n' + '\n'.join(f'#SBATCH --{key}={value}'
                                    for key, value in directives.items())
        # Reuse the template environment setup, replacing its dispatch/launch block.
        setup = body.split('if [[ "${1:-}" == "--combine" ]]; then', 1)[0]
        launch = (shlex.join(['python', *extra]) if extra
                  else f'mpirun -np {ranks} ./hasegawa-wakatani'
                  + (' restart' if name.startswith('DNS') else ''))
        content = (header + '\n\nset -euo pipefail' + setup
                   + 'cd -- ' + shlex.quote(str(run_dir)) + '\n' + launch + '\n')
        print(f'Submission script: {script}\n{content}', flush=True)
        if args.dry_run:
            return name
        script.write_text(content)
        script.chmod(0o755)
        job = subprocess.check_output(['sbatch', '--parsable', script.name],
                                      cwd=run_dir, text=True).strip().split(';')[0]
        if not job.isdigit():
            raise RuntimeError(f'Unexpected sbatch job ID: {job!r}')
        print(f'Submitted {name}: {job}', flush=True)
        with (ROOT / 'hw_submissions.log').open('a') as log:
            log.write(f'{name} {job} {run_dir}\n')
        return job

    for model, modified, fine, factor, coarse, run_dir, checkpoint, source in plans:
        if factor == 1:
            preparation = submit(run_dir, f'prepare{model}{fine}', 32, 1,
                extra=[str(TEMPLATE / 'prepare_dataset_restart.py'), str(run_dir),
                       str(source), '--fine-n', str(fine)])
            submit(run_dir, f'DNS{model}{fine}', 32, 1, preparation)
        else:
            submit(run_dir, f'LES{model}{fine}_{coarse}', 1, 32)


if __name__ == '__main__':
    main()
