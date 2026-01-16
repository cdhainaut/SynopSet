from __future__ import annotations
import argparse
from ..io import open_normalize
from ..plotting import animate_map


def main():
    ap = argparse.ArgumentParser(description="Quick map viewer with cluster overlay.")
    ap.add_argument(
        "grib", type=str, help="Path to GRIB/NetCDF (with optional cluster_id)"
    )
    ap.add_argument("--var", type=str, default="u10")
    ap.add_argument("--every", type=int, default=1, help="Frame stride")
    args = ap.parse_args()

    ds = open_normalize(args.grib)
    animate_map(ds, var=args.var, every=args.every)


if __name__ == "__main__":
    main()
