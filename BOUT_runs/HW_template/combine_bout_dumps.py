#!/usr/bin/env python3
"""Combine BOUT.dmp.<rank>.nc files, removing MPI and boundary guard cells."""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datadir", nargs="?", default="data",
                        help="Dump directory (default: data)")
    parser.add_argument("--output", default="BOUT.dmp.nc",
                        help="Output filename within datadir (default: BOUT.dmp.nc)")
    parser.add_argument("--snapshot", type=Path,
                        help="Export one guard-free NPZ snapshot (U=n, V=phi, P=vort) instead")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--time", type=float, help="Exact DNS output time to export")
    selection.add_argument("--time-index", type=int, default=-1,
                        help="Snapshot output index (default: final output)")
    args = parser.parse_args()
    datadir = Path(args.datadir).resolve()
    if Path(args.output).name != args.output:
        parser.error("--output must be a filename, not a path")
    output = datadir / args.output
    if not args.snapshot and output.exists():
        parser.error(f"Output already exists: {output}")
    dumps = [p for p in datadir.glob("BOUT.dmp.*.nc")
             if p.name.split(".")[2].isdigit()]
    if not dumps:
        parser.error(f"No BOUT.dmp.<rank>.nc files found in {datadir}")

    if args.snapshot:
        import numpy as np
        from netCDF4 import Dataset
        from boutdata import collect

        if args.snapshot.exists():
            parser.error(f"Output already exists: {args.snapshot}")
        with Dataset(datadir / "BOUT.dmp.0.nc") as dump:
            nranks = int(dump["NXPE"][...]) * int(dump["NYPE"][...])
            times = np.asarray(dump["t_array"][:]).reshape(-1)
        missing = [rank for rank in range(nranks)
                   if not (datadir / f"BOUT.dmp.{rank}.nc").is_file()]
        if missing:
            parser.error(f"Missing MPI dump ranks: {missing}")
        if args.time is not None:
            matches = np.flatnonzero(np.isclose(times, args.time, rtol=0, atol=1e-8))
            if len(matches) != 1:
                parser.error(f"Expected one output at t={args.time}, found {len(matches)}")
            index = int(matches[0])
        else:
            index = args.time_index
        if index < 0:
            index += len(times)
        if not 0 <= index < len(times):
            parser.error("Snapshot time index is out of range")
        fields = {}
        for key, name in (("U", "n"), ("V", "phi"), ("P", "vort")):
            values = np.asarray(collect(name, path=str(datadir),
                                tind=[index, index], xguards=False,
                                yguards=False, strict=True, info=False))
            if values.ndim != 4 or values.shape[0] != 1 or values.shape[2] != 1:
                parser.error(f"Expected (1, x, 1, z) for {name}, got {values.shape}")
            field = values[0, :, 0, :]
            if field.shape[0] != field.shape[1] or not np.isfinite(field).all():
                parser.error(f"Invalid square finite field: {name}")
            fields[key] = field
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        with args.snapshot.open("xb") as stream:
            np.savez_compressed(stream, **fields, time=times[index], simtime=times[index])
        print(f"Exported DNS output {index} (t={times[index]}) to {args.snapshot}")
        return

    from boutdata.squashoutput import squashoutput

    squashoutput(datadir=str(datadir), outputname=args.output,
                 xguards=False, yguards=False, delete=False)
    print(f"Combined {len(dumps)} dump files into {output}")


if __name__ == "__main__":
    main()
