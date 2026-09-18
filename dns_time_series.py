"""Load and interpolate saved DNS fields at physical simulation times."""

from pathlib import Path

import numpy as np


class DNSTimeSeries:
    """Interpolate U, V, P snapshots, returning (batch, 3, n, n) arrays."""

    def __init__(self, directory, fine_n, dtype=np.float64):
        self.directory = Path(directory).expanduser()
        self.fine_n = fine_n
        self.dtype = np.dtype(dtype)
        if not np.issubdtype(self.dtype, np.floating):
            raise ValueError("DNS interpolation requires a floating-point dtype")
        snapshots = []
        for path in sorted(self.directory.glob("*.npz")):
            with np.load(path, allow_pickle=False) as snapshot:
                if "simtime" not in snapshot:
                    raise ValueError(f"{path}: missing simtime")
                timestamp = np.asarray(snapshot["simtime"])
                if timestamp.ndim != 0 or not np.isfinite(timestamp):
                    raise ValueError(f"{path}: simtime must be a finite scalar")
                snapshots.append((float(timestamp), path))
        if not snapshots:
            raise FileNotFoundError(f"No DNS .npz snapshots found in {self.directory}")
        snapshots.sort(key=lambda item: item[0])
        self.times = np.asarray([time for time, _ in snapshots])
        self.paths = [path for _, path in snapshots]
        if np.any(np.diff(self.times) <= 0):
            raise ValueError("DNS snapshots must have distinct simulation times")
        self._cache = {}
        self._shape = None

    def _load(self, index):
        if index not in self._cache:
            path = self.paths[index]
            with np.load(path, allow_pickle=False) as snapshot:
                missing = [name for name in ("U", "V", "P") if name not in snapshot]
                if missing:
                    raise ValueError(f"{path}: missing fields {missing}")
                planes = [np.asarray(snapshot[name], dtype=self.dtype)
                          for name in ("U", "V", "P")]
                planes = [plane[None] if plane.ndim == 2 else plane for plane in planes]
                for plane in planes:
                    if (plane.ndim != 3 or plane.shape[0] < 1
                            or plane.shape[-2:] != (self.fine_n, self.fine_n)):
                        raise ValueError(f"{path}: invalid DNS field shape {plane.shape}")
                fields = np.stack(planes, axis=1)
            if not np.all(np.isfinite(fields)):
                raise ValueError(f"{path}: DNS fields contain NaN or Inf")
            if self._shape is not None and fields.shape != self._shape:
                raise ValueError(f"{path}: DNS snapshot shape changed")
            self._shape = fields.shape
            self._cache[index] = fields
        return self._cache[index]

    def interpolate(self, simtime):
        """Linearly interpolate within the saved time range without extrapolation."""
        time = float(simtime)
        if not np.isfinite(time) or time < self.times[0] or time > self.times[-1]:
            raise ValueError(
                f"DNS time {time} is outside [{self.times[0]}, {self.times[-1]}]"
            )
        upper = int(np.searchsorted(self.times, time))
        indices = (upper,) if self.times[upper] == time else (upper - 1, upper)
        # Retain only the snapshots needed for this call, including backward requests.
        self._cache = {index: fields for index, fields in self._cache.items()
                       if index in indices}
        if len(indices) == 1:
            return self._load(upper).copy()
        lower = upper - 1
        weight = (time - self.times[lower]) / (self.times[upper] - self.times[lower])
        return ((1.0 - weight) * self._load(lower) + weight * self._load(upper)).astype(
            self.dtype, copy=False
        )
