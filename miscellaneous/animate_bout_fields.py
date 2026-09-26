#!/usr/bin/env python3
"""Animate n, phi and vort from MPI BOUT++ dumps (requires boutdata,
netCDF4, numpy, matplotlib and pillow; MP4 output also requires ffmpeg).

Example: python animate_bout_fields.py --step 2 --output bout_fields.gif
The selected frames are loaded into memory; use --step to reduce memory use.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import numpy as np
from netCDF4 import Dataset
from boutdata import collect


DEFAULT_DATA = (Path(__file__).resolve().parent.parent / "BOUT_repo/build_release"
                / "examples/hasegawa-wakatani/data_DNS_restart32MPI")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datadir", nargs="?", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=Path("bout_fields.gif"))
    parser.add_argument("--start", type=int, default=0, help="First frame index")
    parser.add_argument("--stop", type=int, help="Exclusive final frame index")
    parser.add_argument("--step", type=int, default=1, help="Frame stride")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()
    if args.step < 1 or args.fps < 1 or args.start < 0:
        parser.error("step and fps must be positive; start must be nonnegative")
    if args.output.suffix.lower() not in (".gif", ".mp4"):
        parser.error("output must end in .gif or .mp4")
    if args.output.suffix.lower() == ".mp4" and not FFMpegWriter.isAvailable():
        parser.error("ffmpeg is unavailable; use .gif or install ffmpeg")

    with Dataset(args.datadir / "BOUT.dmp.0.nc") as dump:
        nranks = int(dump["NXPE"][...]) * int(dump["NYPE"][...])
        times = np.asarray(dump["t_array"][:]).reshape(-1)
    missing = [rank for rank in range(nranks)
               if not (args.datadir / f"BOUT.dmp.{rank}.nc").is_file()]
    if missing:
        parser.error(f"Missing MPI dump ranks: {missing}")
    selection = slice(args.start, args.stop, args.step)
    times = times[selection]
    if not len(times):
        parser.error("The requested frame selection is empty")

    fields = []
    for name in ("n", "phi", "vort"):
        print(f"Collecting {name} from {nranks} MPI dumps...", flush=True)
        data = np.asarray(collect(name, path=str(args.datadir), tind=selection,
                                  xguards=False, yguards=False, info=False,
                                  strict=True))
        if data.ndim != 4 or data.shape[2] != 1:
            parser.error(f"Expected (t, x, 1, z) for {name}, got {data.shape}")
        fields.append(data[:, :, 0, :])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    images = []
    for ax, name, data in zip(axes, ("Density (n)", "Potential (phi)",
                                     "Vorticity (vort)"), fields):
        limit = float(np.nanmax(np.abs(data))) or 1.0
        im = ax.imshow(data[0].T, origin="lower", cmap="RdBu_r",
                       vmin=-limit, vmax=limit, interpolation="nearest")
        ax.set(title=name, xlabel="x grid index", ylabel="z grid index")
        fig.colorbar(im, ax=ax, shrink=0.8)
        images.append(im)
    title = fig.suptitle("")

    def update(frame):
        for im, data in zip(images, fields):
            im.set_data(data[frame].T)
        title.set_text(f"BOUT++   t = {times[frame]:.6g}")
        return [*images, title]

    animation = FuncAnimation(fig, update, frames=len(times),
                              interval=1000 / args.fps, blit=False)
    writer = PillowWriter if args.output.suffix.lower() == ".gif" else FFMpegWriter
    args.output.parent.mkdir(parents=True, exist_ok=True)
    animation.save(str(args.output), writer=writer(fps=args.fps), dpi=100)
    plt.close(fig)
    print(f"Saved {len(times)} frames to {args.output.resolve()}")


if __name__ == "__main__":
    main()
