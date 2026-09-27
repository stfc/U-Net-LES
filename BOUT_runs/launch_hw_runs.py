#!/usr/bin/env python3
"""Build and submit HW/mHW DNS256/512 and factor 2/4/8 LES via HW_template/run.sh.

Run from BOUT_runs: python launch_hw_runs.py [--dry-run]
DNS -> snapshot export -> three LES jobs, with Slurm afterok dependencies.
Existing run directories are skipped. No jobs are submitted until all new builds
succeed. Missing LES cases can reuse an existing DNS snapshot; otherwise they
are skipped until that snapshot is available. DNS defaults to t=0..2000; LES uses DNS at t=500 and evolves for 1500 time units.
"""
import argparse
import configparser
from pathlib import Path
import shlex
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
TEMPLATE = ROOT / 'HW_template'
REPO = ROOT.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--dry-run', action='store_true', help='Print plan without writing, building or submitting')
    parser.add_argument('--prefix', default='HW', help='HW run prefix; modified runs prepend m (default: HW and mHW)')
    parser.add_argument('--dns-nout', type=int, default=2000)
    parser.add_argument('--les-nout', type=int, default=1500)
    parser.add_argument('--snapshot-time', type=float, default=500., help='DNS time used for LES (default: 500)')
    args = parser.parse_args()
    if not args.prefix or Path(args.prefix).name != args.prefix or args.prefix in ('.', '..'):
        parser.error('prefix must be a directory name, not a path')
    for value in (args.dns_nout, args.les_nout):
        if value is not None and value <= 0:
            parser.error('nout must be positive')

    if not 0 <= args.snapshot_time <= args.dns_nout or not args.snapshot_time.is_integer():
        parser.error('snapshot-time must be an integer output time between 0 and dns-nout')

    plans = []
    for model, modified in (('HW', False), ('mHW', True)):
        prefix = f'm{args.prefix}' if modified else args.prefix
        for fine in (256, 512):
            dns_dir = ROOT / f'{prefix}_DNS{fine}'
            snapshot = dns_dir / 'data' / 'les_initial.npz'
            for factor in (1, 2, 4, 8):
                coarse = fine // factor
                run_dir = dns_dir if factor == 1 else ROOT / f'{prefix}_DNS{fine}_LES{coarse}'
                if run_dir.exists():
                    print(f'Skipping existing case: {run_dir.name}', flush=True)
                    continue
                if factor != 1 and dns_dir.exists() and not snapshot.is_file():
                    print(f'Skipping {run_dir.name}: existing DNS snapshot is not ready: {snapshot}', flush=True)
                    continue
                checkpoint = None if factor == 1 else (
                    REPO / f'training/training_{model}_dns{fine}_les{coarse}/checkpoints/best.pt'
                )
                source = None if factor == 1 else snapshot
                if checkpoint is not None and not checkpoint.is_file():
                    parser.error(f'Missing checkpoint: {checkpoint}')
                plans.append((model, modified, fine, factor, coarse, run_dir, checkpoint, source))
    if not plans:
        print('No new cases to launch.')
        return
    if not args.dry_run and not shutil.which('sbatch'):
        parser.error('sbatch is not available')

    for model, modified, fine, factor, coarse, run_dir, checkpoint, source in plans:
        print(f'{run_dir.name}: grid={coarse}, kappa={1.0 if modified else fine / 512:g}, modified={str(modified).lower()}, dns_init={-1 if factor == 1 else 0}, ranks={(128 if fine == 512 else 32) if factor == 1 else 1}', flush=True)
        print(f'  checkpoint={checkpoint}\n  dns_path={source}', flush=True)
        if args.dry_run:
            continue
        (run_dir / 'data').mkdir(parents=True)
        # A synthetic section preserves BOUT's top-level options with ConfigParser.
        config = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=('#',))
        config.optionxform = str
        config.read_string('[root]\n' + (TEMPLATE / 'data/BOUT.inp').read_text())
        config['mesh'].update(nx=str(coarse + 4), nz=str(coarse))
        # Leave decomposition to BOUT++, including when the template sets it.
        for section in config.values():
            for key in list(section):
                if key.upper() in {'NXPE', 'NYPE', 'NZPE'}:
                    del section[key]
        config['root']['timestep'] = '1'
        nout = args.dns_nout if factor == 1 else args.les_nout
        if nout is not None:
            config['root']['nout'] = str(nout)
        config['hw']['kappa'] = str(1.0 if modified else fine / 512)
        config['hw']['modified'] = str(modified).lower()
        config['unet'].update(dns_init=str(-1 if factor == 1 else 0),
                             device='cpu' if factor == 1 else 'cuda',
                             write_diagnostics='0' if factor == 1 else '2',
                             write_arakawa_terms='false')
        if factor == 1:
            config['unet'].pop('checkpoint', None)
            config['unet'].pop('dns_path', None)
        else:
            config['unet'].update(checkpoint=str(checkpoint), dns_path=str(source))
        import io
        output = io.StringIO()
        config.write(output)
        (run_dir / 'data/BOUT.inp').write_text(output.getvalue().removeprefix('[root]\n'))
        subprocess.run(['bash', str(TEMPLATE / 'build.sh'), str(run_dir)], check=True)
        (run_dir / 'hasegawa-wakatani').symlink_to('build_release/hasegawa-wakatani')

    def submit(run_dir, name, ranks, cpus, dependency=None, extra=(), nodes=1):
        # Save the complete submission (resources, dependency and command) locally.
        script = run_dir / ('export.sh' if extra else 'run.sh')
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
        launch = (shlex.join(['python', *extra[1:]]) if extra
                  else f'mpirun -np {ranks} ./hasegawa-wakatani')
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

    exports = {}
    for model, modified, fine, factor, coarse, run_dir, checkpoint, source in plans:
        if factor == 1:
            dns = submit(run_dir, f'DNS{model}{fine}', 128 if fine == 512 else 32, 1,
                         nodes=2 if fine == 512 else 1)
            exports[model, fine] = submit(run_dir, f'export{model}{fine}', 1, 1, dns,
                ['--combine', str(TEMPLATE / 'combine_bout_dumps.py'), str(run_dir / 'data'),
                 '--snapshot', str(run_dir / 'data/les_initial.npz'), '--time', str(args.snapshot_time)])
        else:
            submit(run_dir, f'LES{model}{fine}_{coarse}', 1, 32, exports.get((model, fine)))


if __name__ == '__main__':
    main()
