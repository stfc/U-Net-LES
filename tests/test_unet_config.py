import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from unet_config import load_unet_config, parse_bool


class UnetConfigTests(unittest.TestCase):
    def test_input_values_comments_booleans_and_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'BOUT.inp'
            path.write_text('''timestep = 1
[hw]
alpha = 1
[unet]
implicit = false # C++ and Python must agree
omega_from_potential = yes
alpha = 0.25
checkpoint = "/path with spaces/best.pt"
site_packages =
post_cutoff_fraction = auto
inverse_cutoff_fraction = none
''')
            values, sources = load_unet_config(path)
            self.assertEqual(values['implicit'], '0')
            self.assertEqual(values['omega_from_potential'], '1')
            self.assertEqual(values['alpha'], '0.25')
            self.assertEqual(values['checkpoint'], '/path with spaces/best.pt')
            self.assertEqual(values['site_packages'], '')
            self.assertEqual(values['post_cutoff_fraction'], 'auto')
            self.assertEqual(values['inverse_cutoff_fraction'], 'none')
            self.assertEqual(values['bout_inp'], str(path))
            self.assertIn('[unet]', sources['alpha'])

    def test_external_environment_is_ignored_and_unchanged(self):
        with patch.dict(os.environ, {
            'PBOUT_UNET_CHECKPOINT': '/wrong/checkpoint.pt',
            'PBOUT_UNET_DEVICE': 'wrong-device',
            'PBOUT_UNET_BOUT_INP': '/missing/input',
            'PBOUT_UNET_IMPLICIT': 'invalid',
        }):
            before = dict(os.environ)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'BOUT.inp'
                path.write_text('[unet]\nimplicit = true\n')
                values, _ = load_unet_config(path)
                self.assertEqual(values['implicit'], '1')
                self.assertNotIn('checkpoint', values)
                self.assertNotIn('device', values)
            embedded, _ = load_unet_config(embedded={'implicit': '0', 'device': 'cpu'})
            self.assertEqual(embedded, {'implicit': '0', 'device': 'cpu'})
            self.assertEqual(dict(os.environ), before)

    def test_embedded_resolved_options_do_not_reread_file(self):
        values, _ = load_unet_config('/nonexistent', embedded={
            '__name__': '_pbout_unet_options', 'implicit': 'false',
            'bout_inp': 'other/data/custom.inp', 'dns_init': '3',
        })
        self.assertEqual(values['dns_init'], '3')
        self.assertEqual(values['implicit'], '0')
        self.assertNotIn('__name__', values)

    def test_boolean_validation(self):
        for value in ('true', 'TRUE', 'yes', 'on', '1'):
            self.assertTrue(parse_bool(value, 'implicit'))
        for value in ('false', 'FALSE', 'no', 'off', '0'):
            self.assertFalse(parse_bool(value, 'implicit'))
        for value in ('', '3', 'maybe'):
            with self.assertRaisesRegex(ValueError, 'implicit'):
                load_unet_config(embedded={'implicit': value})

    def test_missing_file_fails_instead_of_using_environment(self):
        with self.assertRaises(FileNotFoundError):
            load_unet_config('/nonexistent/BOUT.inp')


if __name__ == '__main__':
    unittest.main()
