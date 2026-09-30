"""Run with the BOUT Python environment: python -m unittest discover -s tests -p test_launch_hw_runs.py."""

import ast
import contextlib
import io
from pathlib import Path
import random
import tempfile
import sys
import unittest
from unittest.mock import patch

import numpy as np
from netCDF4 import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'BOUT_runs'))

import launch_hw_runs as launcher
from HW_template import prepare_dataset_restart as restart_builder


class LaunchTests(unittest.TestCase):
    def test_split_matches_training(self):
        # Execute the actual training split function without loading Torch/CUDA.
        tree = ast.parse((launcher.REPO / 'train_germano_unet.py').read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'split_files')
        namespace = {'random': random,
                     'recursive_npz_files': lambda d: sorted(p for p in d.rglob('*.npz') if p.is_file())}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<training split>', 'exec'), namespace)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index in range(103):
                (root / f'field{index}.npz').touch()
            for mode in ('random', 'sequential'):
                parameters = dict(validation_fraction=0.1, test_fraction=0.1, split_mode=mode, seed=7)
                training, validation, testing = namespace['split_files'](root, **parameters)
                self.assertEqual(launcher.test_files(root, parameters), testing)
                self.assertFalse(set(testing) & (set(training) | set(validation)))
            for split in ('train', 'validation', 'test'):
                (root / split).mkdir()
                (root / split / 'field.npz').touch()
            self.assertEqual(launcher.test_files(root, parameters), [root / 'test/field.npz'])

    def test_restart_fields_and_periodic_guards(self):
        for fine_n in (256, 512):
            with self.subTest(fine_n=fine_n), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                data = root / 'data'
                data.mkdir()
                (data / 'BOUT.inp').write_text('NYPE = 1\n' + (launcher.TEMPLATE / 'data/BOUT.inp').read_text())
                # A failed attempt must not prevent preparing fresh restart files.
                (root / 'restart_seed').mkdir()
                (root / 'restart_seed' / 'failure.log').write_text('original failure')
                angle = np.arange(fine_n) * (2 * np.pi / fine_n)
                potential = np.sin(angle)[:, None] + np.cos(angle)[None, :]
                density = potential + 3
                source = root / 'field.npz'
                np.savez(source, U=density, V=potential + 7, P=np.zeros_like(potential))
                width = fine_n // 32

                def native_zero_step(command, **kwargs):
                    self.assertEqual(command[:3], ['mpirun', '-np', '32'])
                    seed = Path(command[-1])
                    self.assertIn('nout = 0', (seed / 'BOUT.inp').read_text())
                    self.assertNotIn('NYPE', (seed / 'BOUT.inp').read_text())
                    self.assertIn('NXPE = 32', (seed / 'BOUT.inp').read_text())
                    for rank in range(32):
                        with Dataset(seed / f'BOUT.restart.{rank}.nc', 'w') as output:
                            for name, size in (('x', width + 4), ('y', 1), ('z', fine_n)):
                                output.createDimension(name, size)
                            for name, value in (('NXPE', 32), ('NYPE', 1), ('tt', 0)):
                                output.createVariable(name, 'i4')[...] = value
                            for name in ('n', 'vort'):
                                output.createVariable(name, 'f8', ('x', 'y', 'z'))[:] = 0

                with patch.object(restart_builder.subprocess, 'run', side_effect=native_zero_step):
                    restart_builder.prepare(root, source, fine_n)
                expected_vort = -4 * np.sin(np.pi / fine_n)**2 / (50.176 / fine_n)**2 * potential
                interiors = []
                for rank in range(32):
                    indices = np.arange(rank * width - 2, (rank + 1) * width + 2) % fine_n
                    with Dataset(data / f'BOUT.restart.{rank}.nc') as result:
                        np.testing.assert_allclose(result['n'][:, 0, :], potential[indices], atol=1e-12)
                        np.testing.assert_allclose(result['vort'][:, 0, :], expected_vort[indices], atol=1e-11)
                        interiors.append(result['n'][2:-2, 0, :])
                np.testing.assert_allclose(np.concatenate(interiors), potential, atol=1e-12)
                self.assertEqual((root / 'restart_seed/failure.log').read_text(), 'original failure')
                with self.assertRaises(FileExistsError):
                    restart_builder.prepare(root, source, fine_n)

    def test_job_dependencies_and_resources(self):
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, patch.object(launcher, 'ROOT', Path(directory)), \
                patch.object(launcher, 'select_test_field', return_value=Path('/test/field.npz')), \
                patch('sys.argv', ['launch_hw_runs.py', '--dry-run']), contextlib.redirect_stdout(output):
            launcher.main()
        scripts = output.getvalue().split('Submission script: ')[1:]
        self.assertEqual(len(scripts), 20)
        for script in scripts:
            self.assertNotIn('--time 200', script)
            self.assertNotIn('--time 500', script)
            if '/prepare_restart.sh' in script:
                self.assertIn('#SBATCH --ntasks=32\n', script)
                self.assertIn('prepare_dataset_restart.py', script)
            elif '#SBATCH --job-name=DNS' in script:
                self.assertIn('#SBATCH --dependency=afterok:prepare', script)
                self.assertIn('mpirun -np 32 ./hasegawa-wakatani restart', script)
            else:
                self.assertNotIn('#SBATCH --dependency', script)
                self.assertIn('#SBATCH --cpus-per-task=32', script)


if __name__ == '__main__':
    unittest.main()
