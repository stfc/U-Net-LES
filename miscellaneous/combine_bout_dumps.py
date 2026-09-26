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
    args = parser.parse_args()
    datadir = Path(args.datadir).resolve()
    if Path(args.output).name != args.output:
        parser.error("--output must be a filename, not a path")
    output = datadir / args.output
    if output.exists():
        parser.error(f"Output already exists: {output}")
    dumps = [p for p in datadir.glob("BOUT.dmp.*.nc")
             if p.name.split(".")[2].isdigit()]
    if not dumps:
        parser.error(f"No BOUT.dmp.<rank>.nc files found in {datadir}")

    from boutdata.squashoutput import squashoutput

    squashoutput(datadir=str(datadir), outputname=args.output,
                 xguards=False, yguards=False, delete=False)
    print(f"Combined {len(dumps)} dump files into {output}")


if __name__ == "__main__":
    main()
