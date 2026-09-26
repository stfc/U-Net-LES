"""Fixed-value x / periodic y operators on cell-centred grids.

The x faces lie half a cell outside the stored samples. Odd reflection of
``field - boundary_value`` imposes Dirichlet data there. A sine transform in
x and Fourier transform in y diagonalize the differential Germano filter.
"""
import math

import numpy as np
import torch


def sine_transform(field):
    """Orthonormal DST-II along x, implemented using a length-2N FFT."""
    n = field.shape[-2]
    extended = torch.cat((field, -field.flip(-2)), dim=-2)
    modes = torch.arange(1, n + 1, device=field.device, dtype=field.dtype)
    phase = torch.exp(-0.5j * math.pi * modes / n)[:, None]
    coefficients = (1j * torch.fft.fft(extended, dim=-2)[..., 1:n + 1, :] * phase).real
    weights = torch.ones_like(modes) / math.sqrt(2 * n)
    weights[-1] /= math.sqrt(2)
    return coefficients * weights[:, None]


def inverse_sine_transform(coefficients):
    n = coefficients.shape[-2]
    modes = torch.arange(1, n + 1, device=coefficients.device, dtype=coefficients.dtype)
    weights = torch.ones_like(modes) * math.sqrt(2 * n)
    weights[-1] *= math.sqrt(2)
    phase = -1j * torch.exp(0.5j * math.pi * modes / n)
    positive = coefficients * (weights * phase)[:, None]
    spectrum = torch.cat((torch.zeros_like(positive[..., :1, :]), positive), dim=-2)
    return torch.fft.irfft(spectrum, n=2 * n, dim=-2)[..., :n, :]


def transform(field):
    return torch.fft.fft(sine_transform(field), dim=-1, norm="ortho")


def inverse_transform(coefficients):
    return inverse_sine_transform(torch.fft.ifft(coefficients, dim=-1, norm="ortho").real)


def wavenumbers(field, lx, ly):
    nx, ny = field.shape[-2:]
    kx = math.pi * torch.arange(1, nx + 1, device=field.device, dtype=field.dtype) / lx
    ky = 2 * math.pi * torch.fft.fftfreq(ny, d=ly / ny, device=field.device, dtype=field.dtype)
    return kx[:, None].square() + ky[None, :].square()


def apply_multiplier(field, multiplier, boundary_value=0.0):
    return inverse_transform(transform(field - boundary_value) * multiplier) + boundary_value


def filter_field(field, alpha, lx, ly, boundary_value=0.0):
    return apply_multiplier(field, 1 / (1 + alpha**2 * wavenumbers(field, lx, ly)), boundary_value)


def cutoff(field, lx, ly, fraction, order=8, boundary_value=0.0, gaussian=False):
    if fraction is None:
        return field
    if not 0 < fraction <= 1 or order <= 0:
        raise ValueError("Cutoff fraction must be in (0, 1] and order positive")
    nx, ny = field.shape[-2:]
    kc = fraction * min(math.pi * nx / lx, math.pi * ny / ly)
    k2 = wavenumbers(field, lx, ly)
    multiplier = torch.exp(-0.5 * k2 / kc**2) if gaussian else torch.exp(-(k2.sqrt() / kc)**order)
    return apply_multiplier(field, multiplier, boundary_value)


def reconstruct(field, alpha, lx, ly, method="exact", iterations=1,
                relaxation=1.0, boundary_value=0.0):
    if method == "exact":
        return apply_multiplier(field, 1 + alpha**2 * wavenumbers(field, lx, ly), boundary_value)
    if method != "van_cittert":
        raise ValueError(f"Unknown inverse method {method!r}")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 0:
        raise ValueError("iterations must be a nonnegative integer")
    if not math.isfinite(relaxation) or not 0 < relaxation <= 1:
        raise ValueError("relaxation must be in (0, 1]")
    estimate = field
    for _ in range(iterations):
        estimate = estimate + relaxation * (field - filter_field(estimate, alpha, lx, ly, boundary_value))
    # A Dirichlet field has no freely adjustable constant mode.
    return estimate


def resize(field, target_n, boundary_value=0.0):
    """Sine/Fourier resampling with cell-centred x sample positions."""
    nx, ny = field.shape[-2:]
    coefficients = sine_transform(field - boundary_value)
    target = coefficients.new_zeros((*field.shape[:-2], target_n, ny))
    count = min(nx, target_n)
    target[..., :count, :] = coefficients[..., :count, :]
    # The highest DST-II mode has twice the squared norm of other modes.
    if target_n > nx:
        target[..., nx - 1, :] *= math.sqrt(2)
    elif target_n < nx:
        target[..., target_n - 1, :] /= math.sqrt(2)
    target *= math.sqrt(target_n / nx)
    x_resized = inverse_sine_transform(target)
    spectrum = torch.fft.rfft(x_resized, dim=-1)
    resized = spectrum.new_zeros((*x_resized.shape[:-1], target_n // 2 + 1))
    count = min(ny, target_n)
    resized[..., :count // 2 + 1] = spectrum[..., :count // 2 + 1]
    if count % 2 == 0:
        if target_n < ny:
            resized[..., count // 2] = 2 * resized[..., count // 2].real
        elif target_n > ny:
            resized[..., count // 2] *= 0.5
    return torch.fft.irfft(resized, n=target_n, dim=-1) * (target_n / ny) + boundary_value


def shift(field, i, j=0, boundary_value=0.0):
    """torch.roll convention, using odd x ghost cells at fixed-value faces."""
    if i:
        radius = abs(i)
        if radius > field.shape[-2]:
            raise ValueError("Ghost stencil exceeds x grid size")
        extended = torch.cat((
            2 * boundary_value - field[..., :radius, :].flip(-2), field,
            2 * boundary_value - field[..., -radius:, :].flip(-2),
        ), dim=-2)
        field = extended[..., radius - i:radius - i + field.shape[-2], :]
    return torch.roll(field, j, -1)


def laplacian(field, lx, ly, order=2, boundary_value=0.0):
    dx, dy = lx / field.shape[-2], ly / field.shape[-1]
    if order == 2:
        return ((shift(field, 1, boundary_value=boundary_value) - 2 * field
                 + shift(field, -1, boundary_value=boundary_value)) / dx**2
                + (shift(field, 0, 1) - 2 * field + shift(field, 0, -1)) / dy**2)
    if order != 4 or min(field.shape[-2:]) < 5:
        raise ValueError("Fourth-order Laplacian requires at least 5 points per direction")
    result = torch.zeros_like(field)
    for axis, spacing in ((0, dx), (1, dy)):
        def neighbor(offset):
            return shift(field, offset if axis == 0 else 0,
                         offset if axis == 1 else 0, boundary_value)
        result = result + (-neighbor(2) + 16 * neighbor(1) - 30 * field
                           + 16 * neighbor(-1) - neighbor(-2)) / (12 * spacing**2)
    return result


def shell_indices(field, lx, ly):
    dk = min(math.pi / lx, 2 * math.pi / ly)
    return torch.round(wavenumbers(field, lx, ly).sqrt() / dk).long()


def shell_power(field, lx, ly, boundary_value=0.0):
    """Mean unnormalized modal power, (channels, shells), for saved targets."""
    shells = shell_indices(field, lx, ly).flatten()
    power = (transform(field - boundary_value).abs().square()
             * (field.shape[-2] * field.shape[-1])).flatten(-2)
    sums = power.new_zeros((*power.shape[:-1], int(shells.max()) + 1))
    sums.scatter_add_(-1, shells.expand_as(power), power)
    counts = torch.bincount(shells, minlength=sums.shape[-1]).clamp_min(1)
    return sums / counts


def spectrum_filter(field, lx, ly, target, min_wavenumber=1.0, boundary_value=0.0):
    if target is None:
        raise ValueError("The filtered-DNS spectrum filter requires a stored spectrum")
    if not math.isfinite(min_wavenumber) or min_wavenumber < 0:
        raise ValueError("min_wavenumber must be finite and nonnegative")
    target = torch.as_tensor(target, dtype=field.dtype, device=field.device)
    current = shell_power(field, lx, ly, boundary_value).mean(dim=0)
    if target.shape != current.shape:
        raise ValueError("Stored fixed-x spectrum shape does not match the grid/channels")
    scale = (target / current.clamp_min(torch.finfo(field.dtype).tiny)).sqrt().clamp(max=1)
    shells = shell_indices(field, lx, ly)
    multiplier = scale[:, shells][None]
    multiplier = torch.where(wavenumbers(field, lx, ly).sqrt() <= min_wavenumber, 1., multiplier)
    return apply_multiplier(field, multiplier, boundary_value)


def numpy_call(function, field, *args, **kwargs):
    array = np.asarray(field)
    tensor = torch.as_tensor(np.ascontiguousarray(array))
    return function(tensor, *args, **kwargs).detach().numpy().astype(array.dtype, copy=False)
