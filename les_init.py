"""Checkpoint-free snapshot initialization for native BOUT++ LES modes 4-7."""
import sys
from pathlib import Path

try:
    import _pbout_unet_options as _options
    CONFIG = vars(_options)
except ImportError:
    CONFIG = {}
if CONFIG.get('site_packages'):
    sys.path.insert(0, CONFIG['site_packages'])
import numpy as np


def initialize_snapshot(path, coarse_n, dx, dz, alpha=0.25, laplacian_order=2):
    """Center, derive vorticity, Germano filter, and spectrally downsample U,V,P."""
    if coarse_n < 2 or not all(np.isfinite(v) and v > 0 for v in (dx, dz)):
        raise ValueError('LES grid size and spacings must be positive')
    if not np.isfinite(alpha) or alpha < 0:
        raise ValueError('unet:alpha must be finite and nonnegative')
    if laplacian_order not in (2, 4):
        raise ValueError('unet:laplacian_order must be 2 or 4')
    with np.load(path) as snapshot:
        fields = np.stack([snapshot[key] for key in ('U', 'V', 'P')]).astype(float)
    fine_n = fields.shape[-1]
    if fields.shape != (3, fine_n, fine_n) or fine_n < coarse_n:
        raise ValueError('DNS fields must be square and at least as large as the LES grid')
    if not np.isfinite(fields).all():
        raise ValueError('DNS fields contain NaN or Inf')
    lx, lz = dx * coarse_n, dz * coarse_n
    fields[:2] -= fields[:2].mean(axis=(-2, -1), keepdims=True)
    phi = fields[1]
    lap = np.zeros_like(phi)
    for axis, spacing in ((0, lx / fine_n), (1, lz / fine_n)):
        if laplacian_order == 2:
            lap += (np.roll(phi, 1, axis) - 2*phi + np.roll(phi, -1, axis)) / spacing**2
        else:
            lap += (-np.roll(phi, 2, axis) + 16*np.roll(phi, 1, axis)
                    - 30*phi + 16*np.roll(phi, -1, axis)
                    - np.roll(phi, -2, axis)) / (12*spacing**2)
    fields[2] = lap
    kx = 2*np.pi*np.fft.fftfreq(fine_n, d=lx/fine_n)
    kz = 2*np.pi*np.fft.fftfreq(fine_n, d=lz/fine_n)
    transfer = 1 / (1 + alpha**2 * (kx[:, None]**2 + kz[None, :]**2))
    spectrum = np.fft.fftshift(np.fft.fft2(fields) * transfer, axes=(-2, -1))
    start = (fine_n - coarse_n) // 2
    spectrum = spectrum[:, start:start+coarse_n, start:start+coarse_n]
    result = np.fft.ifft2(np.fft.ifftshift(spectrum, axes=(-2, -1))).real
    return np.ascontiguousarray(result * (coarse_n / fine_n)**2, dtype=np.float64)


def initFlow(payload):
    coarse_n = int(round(np.sqrt((len(payload) - 2) / 3)))
    path = CONFIG.get('dns_path')
    if not path:
        raise ValueError('unet:dns_path is required for modes 4-7')
    return initialize_snapshot(Path(path), coarse_n, float(payload[0]), float(payload[1]),
                               float(CONFIG.get('alpha', 0.25)),
                               int(CONFIG.get('laplacian_order', 2))).ravel()
