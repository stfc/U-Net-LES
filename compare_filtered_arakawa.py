#!/usr/bin/env python3
"""Compare Arakawa terms before and after Germano filtering on the DNS grid.

For each nonlinear Hasegawa--Wakatani term this script compares

    J(G(phi), G(q))    and    G(J(phi, q)),

where G is the differential (Germano) filter and q is either vorticity or
density.  Every operation stays at the input DNS resolution; no spectral
downsampling is performed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from germano_common import (
    HW_FIELD_NAMES,
    germano_filter,
    load_checkpoint,
    load_dns_snapshot,
)
from visualization_common import calculate_arakawa_terms


HERE = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT = (
    HERE / "germano_training_output_32_v10" / "checkpoints" / "best.pt"
)
TERM_NAMES = ("J_V_W", "J_V_U")
TERM_TITLES = (r"$J(\phi,\omega)$", r"$J(\phi,n)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="DNS .npz")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=DEFAULT_CHECKPOINT,
        help="Checkpoint supplying the pBOUT Germano parameters",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("filtered_arakawa_comparison"),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device used only to load the checkpoint (default: cpu)",
    )
    return parser.parse_args()


def relative_l2(reference: np.ndarray, candidate: np.ndarray) -> float:
    denominator = np.linalg.norm(reference.ravel())
    error = np.linalg.norm((candidate - reference).ravel())
    return float(error / denominator) if denominator else float(error)


def plot_comparison(
    from_filtered_fields: tuple[np.ndarray, np.ndarray],
    filtered_dns_terms: tuple[np.ndarray, np.ndarray],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(17.0, 10.0), squeeze=False)
    column_labels = (
        r"$J(G(\phi),G(q))$",
        r"$G(J(\phi,q))$",
        "difference",
    )
    for row, title in enumerate(TERM_TITLES):
        first = from_filtered_fields[row]
        second = filtered_dns_terms[row]
        fields = (first, second, first - second)
        # A common symmetric scale makes all three panels directly comparable.
        limit = max(float(np.max(np.abs(field))) for field in fields)
        for column, (field, column_label) in enumerate(
            zip(fields, column_labels)
        ):
            image = axes[row, column].imshow(
                field.T,
                origin="lower",
                cmap="RdBu_r",
                vmin=-limit if limit else None,
                vmax=limit if limit else None,
                aspect="equal",
            )
            axes[row, column].set_title(f"{title}: {column_label}")
            axes[row, column].set_xlabel("x index")
            axes[row, column].set_ylabel("y index")
            fig.colorbar(image, ax=axes[row, column], shrink=0.82)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"DNS snapshot not found: {args.input}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    checkpoint = load_checkpoint(args.checkpoint, torch.device(args.device))
    field_names = tuple(checkpoint["field_names"])
    if field_names != HW_FIELD_NAMES:
        raise ValueError(
            f"Expected checkpoint fields {HW_FIELD_NAMES}, got {field_names}"
        )
    fine_n = int(checkpoint["fine_n"])
    lx = float(checkpoint["lx"])
    ly = float(checkpoint["ly"])
    alpha = float(checkpoint["alpha"])
    precision = checkpoint.get("precision", "float32")
    if precision not in ("float32", "float64"):
        raise ValueError(f"Unsupported checkpoint precision: {precision}")
    dtype = getattr(np, precision)

    dns = load_dns_snapshot(
        args.input, field_names, fine_n, lx, ly, dtype=dtype
    )

    # Filter all DNS fields independently at the same resolution.
    filtered_dns = np.stack(
        [germano_filter(channel, alpha, lx, ly) for channel in dns]
    ).astype(dtype)
    dns_terms = calculate_arakawa_terms(dns, lx, ly)
    from_filtered_fields = calculate_arakawa_terms(filtered_dns, lx, ly)
    filtered_dns_terms = tuple(
        germano_filter(term, alpha, lx, ly) for term in dns_terms
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = args.output_dir / "arakawa_filter_comparison.png"
    plot_comparison(
        from_filtered_fields, filtered_dns_terms, comparison_path
    )

    arrays: dict[str, np.ndarray] = {
        "dns_fields": dns,
        "filtered_dns_fields": filtered_dns,
    }
    metrics = {}
    for name, dns_term, first, second in zip(
        TERM_NAMES, dns_terms, from_filtered_fields, filtered_dns_terms
    ):
        difference = first - second
        arrays[f"dns_{name}"] = dns_term
        arrays[f"from_filtered_fields_{name}"] = first
        arrays[f"filtered_dns_{name}"] = second
        arrays[f"difference_{name}"] = difference
        metrics[name] = {
            "relative_l2": relative_l2(second, first),
            "rmse": float(np.sqrt(np.mean(difference**2))),
            "max_abs": float(np.max(np.abs(difference))),
        }

    results_path = args.output_dir / "arakawa_filter_comparison.npz"
    np.savez_compressed(
        results_path,
        **arrays,
        alpha=np.asarray(alpha),
        lx=np.asarray(lx),
        ly=np.asarray(ly),
        resolution=np.asarray(fine_n),
        field_names=np.asarray(field_names),
        source_file=np.asarray(str(args.input)),
        checkpoint=np.asarray(str(args.checkpoint)),
    )
    metadata = {
        "input": str(args.input),
        "checkpoint": str(args.checkpoint),
        "resolution": fine_n,
        "lx": lx,
        "ly": ly,
        "alpha": alpha,
        "downsampling": False,
        "comparison": "J(G(phi), G(q)) versus G(J(phi, q))",
        "metrics": metrics,
    }
    metadata_path = args.output_dir / "metrics.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(f"Kept DNS resolution: {fine_n} x {fine_n}")
    print(f"Germano parameters: alpha={alpha:g}, Lx={lx:g}, Ly={ly:g}")
    for name in TERM_NAMES:
        values = metrics[name]
        print(
            f"{name}: relative L2={values['relative_l2']:.6e}, "
            f"RMSE={values['rmse']:.6e}, "
            f"max abs={values['max_abs']:.6e}"
        )
    print(f"Saved arrays to {results_path}")
    print(f"Saved comparison plot to {comparison_path}")
    print(f"Saved metrics to {metadata_path}")


if __name__ == "__main__":
    main()
