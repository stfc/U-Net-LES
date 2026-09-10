#!/usr/bin/env python3
"""Check a U-Net training history for common signs of overfitting."""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from germano_common import \
(
    checkpoint_model_class,
    calculate_losses,
    filter_snapshot,
    load_checkpoint,
    load_dns_snapshot,
    recursive_npz_files,
    validate_hw_checkpoint,
)


DEFAULT_DATASET = Path \
(
    "../../data/BOUT_runs/HW_2D/Papers/PoP23/HW_N256/fields"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser \
    (
        description="Compare training and validation losses in training_history.csv."
    )
    parser.add_argument \
    (
        "run",
        type=Path,
        help="Training output directory or a training_history.csv file",
    )
    parser.add_argument \
    (
        "--metric",
        default="total",
        help="Metric suffix to check (default: total)",
    )
    parser.add_argument \
    (
        "--max-gap",
        type=float,
        default=0.25,
        help="Maximum acceptable (validation - train) / train (default: 0.25)",
    )
    parser.add_argument \
    (
        "--max-degradation",
        type=float,
        default=0.10,
        help="Maximum acceptable final validation increase over its best value "
        "(default: 0.10)",
    )
    parser.add_argument \
    (
        "--dataset",
        type=Path,
        default=DEFAULT_DATASET,
        help=f"DNS snapshot directory (default: {DEFAULT_DATASET})",
    )
    parser.add_argument \
    (
        "--checkpoint",
        type=Path,
        help="Checkpoint (default: RUN/checkpoints/best.pt)",
    )
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument \
    (
        "--plot",
        type=Path,
        help="Comparison image path (default: RUN/vorticity_overfit_check.png)",
    )
    return parser.parse_args()


def save_field_comparison(cases, channel: int, field: str, output_path: Path) -> None:
    """Plot DNS, reconstructed, and absolute error for one field by case."""
    nrows = len(cases)
    fig, axes = plt.subplots \
    (
        nrows,
        3,
        figsize=(12, max(2.2 * nrows, 4)),
        squeeze=False,
        constrained_layout=True,
    )
    value_limit = max \
    (
        float(np.max(np.abs(values[channel])))
        for _, dns, reconstructed in cases
        for values in (dns, reconstructed)
    )
    error_limit = max \
    (
        float(np.max(np.abs(dns[channel] - reconstructed[channel])))
        for _, dns, reconstructed in cases
    )
    image = error_image = None
    for row, (name, dns, reconstructed) in enumerate(cases):
        dns_field = dns[channel]
        reconstructed_field = reconstructed[channel]
        error = np.abs(reconstructed_field - dns_field)
        image = axes[row, 0].imshow \
        (
            dns_field, cmap="RdBu_r", vmin=-value_limit,
            vmax=value_limit, origin="lower",
        )
        axes[row, 1].imshow \
        (
            reconstructed_field, cmap="RdBu_r", vmin=-value_limit,
            vmax=value_limit, origin="lower",
        )
        error_image = axes[row, 2].imshow \
        (
            error, cmap="magma", vmin=0, vmax=error_limit, origin="lower"
        )
        axes[row, 0].set_ylabel(name, fontsize=8)
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
    for axis, title in zip \
    (
        axes[0], (f"DNS {field}", f"Reconstructed {field}", "Absolute error")
    ):
        axis.set_title(title)
    fig.colorbar(image, ax=axes[:, :2], shrink=0.7, label=field)
    fig.colorbar(error_image, ax=axes[:, 2], shrink=0.7, label="Absolute error")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def score_random_reconstructions(args: argparse.Namespace, run_dir: Path) -> None:
    """Reconstruct random DNS snapshots and print mean relative-L2 and R2."""
    if args.samples < 1:
        raise ValueError("--samples must be positive")
    checkpoint_path = args.checkpoint or run_dir / "checkpoints" / "best.pt"
    files = recursive_npz_files(args.dataset)
    if len(files) < args.samples:
        raise ValueError \
        (
            f"Requested {args.samples} samples, but only {len(files)} NPZ files "
            f"were found under {args.dataset}"
        )
    selected = random.Random(args.seed).sample(files, args.samples)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = load_checkpoint(checkpoint_path, device)
    validate_hw_checkpoint(checkpoint)
    precision = checkpoint.get("precision", "float32")
    torch_dtype = getattr(torch, precision)
    numpy_dtype = getattr(np, precision)
    fields = tuple(checkpoint["field_names"])
    fine_n = int(checkpoint["fine_n"])
    coarse_n = int(checkpoint["coarse_n"])
    lx, ly = float(checkpoint["lx"]), float(checkpoint["ly"])
    alpha = float(checkpoint["alpha"])
    output_channels = int(checkpoint["model_output_channels"])
    parameters = checkpoint.get("training_parameters", {})
    mean_np = np.asarray(checkpoint["mean"], dtype=numpy_dtype)
    std_np = np.asarray(checkpoint["std"], dtype=numpy_dtype)
    dns_std_np = np.asarray(checkpoint["dns_std"], dtype=numpy_dtype)
    mean = torch.from_numpy(mean_np).to(device)
    std = torch.from_numpy(std_np).to(device)
    dns_std = torch.from_numpy(dns_std_np).to(device)
    shell_power = checkpoint.get("filtered_dns_shell_power")
    if shell_power is not None:
        shell_power = torch.as_tensor(shell_power, dtype=torch_dtype, device=device)

    model = checkpoint_model_class(checkpoint) \
    (
        len(fields), int(checkpoint["base_channels"]), fine_n // coarse_n,
        output_channels,
        variational=checkpoint.get("variational", False),
    ).to(device=device, dtype=torch_dtype)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    relative_l2, r2, plot_cases = [], [], []
    print(f"\nRandom DNS reconstruction ({args.samples} snapshots, seed={args.seed})")
    print(f"Checkpoint: {checkpoint_path}")
    with torch.no_grad():
        for path in selected:
            dns_np = load_dns_snapshot(path, fields, fine_n, lx, ly, numpy_dtype)
            coarse_np, fine_np = filter_snapshot \
            (
                dns_np, alpha, lx, ly, coarse_n, numpy_dtype
            )
            coarse = torch.from_numpy \
            (
                ((coarse_np - mean_np[0]) / std_np[0])[None]
            ).to(device=device, dtype=torch_dtype)
            fine = torch.from_numpy \
            (
                ((fine_np - mean_np[0]) / std_np[0])[None]
            ).to(device=device, dtype=torch_dtype)
            dns = torch.from_numpy(dns_np[None]).to(device)
            _, _, _, reconstructed = calculate_losses \
            (
                model, coarse, fine, dns, mean, std, dns_std, alpha, lx, ly,
                1.0, checkpoint["vorticity_mode"] == "laplacian_of_potential",
                checkpoint.get("inverse_cutoff_fraction"),
                checkpoint.get("inverse_cutoff_order", 8.0),
                checkpoint.get("post_cutoff_fraction"),
                checkpoint.get("post_cutoff_order", 8.0),
                checkpoint.get("post_filter", parameters.get("post_filter", "high_order")),
                checkpoint.get("inverse_method", parameters.get("inverse_method", "exact")),
                checkpoint.get("van_cittert_iterations", 1),
                checkpoint.get("van_cittert_relaxation", 1.0),
                checkpoint.get("inverse_filter", parameters.get("inverse_filter", "cutoff")),
                shell_power,
                inverse_min_wavenumber=checkpoint.get("inverse_min_wavenumber", 1.0),
            )
            error = reconstructed - dns
            rel = torch.linalg.vector_norm(error.flatten(1), dim=1) / torch.linalg.vector_norm \
            (
                dns.flatten(1), dim=1
            ).clamp_min(1e-15)
            centered = dns - dns.mean(dim=(-2, -1), keepdim=True)
            score = 1.0 - error.square().sum() / centered.square().sum().clamp_min(1e-15)
            relative_l2.append(float(rel.item()))
            r2.append(float(score.item()))
            plot_cases.append \
            (
                (
                    path.name,
                    dns[0].cpu().numpy(),
                    reconstructed[0].cpu().numpy(),
                )
            )
            print(f"  {path.name}: relative L2={relative_l2[-1]:.4f}, R2={r2[-1]:.4f}")
    print(f"Mean reconstruction relative L2: {np.mean(relative_l2):.4f}")
    print(f"Mean reconstruction score (R2):  {np.mean(r2):.4f}")
    plot_path = args.plot or run_dir / "vorticity_overfit_check.png"
    comparison_paths = \
    {
        "U": plot_path.with_name("U_overfit_check.png"),
        "V": plot_path.with_name("V_overfit_check.png"),
        "vorticity": plot_path,
    }
    for channel, (field, output_path) in enumerate(comparison_paths.items()):
        save_field_comparison(plot_cases, channel, field, output_path)
        print(f"{field} comparison: {output_path}")


def main() -> int:
    args = parse_args()
    history_path = args.run / "training_history.csv" if args.run.is_dir() else args.run
    run_dir = args.run if args.run.is_dir() else args.run.parent
    train_key = f"train_{args.metric}"
    validation_key = f"validation_{args.metric}"

    with history_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"No epochs found in {history_path}")
    missing = {"epoch", train_key, validation_key} - set(rows[0])
    if missing:
        raise ValueError \
        (
            f"Missing columns {sorted(missing)}. Available: {sorted(rows[0])}"
        )

    epochs = [int(row["epoch"]) for row in rows]
    train = [float(row[train_key]) for row in rows]
    validation = [float(row[validation_key]) for row in rows]
    best_index = min(range(len(rows)), key=validation.__getitem__)
    best_validation = validation[best_index]
    final_gap = max(0.0, validation[-1] - train[-1]) / max(abs(train[-1]), 1e-15)
    degradation = max(0.0, validation[-1] - best_validation) / max \
    (
        abs(best_validation), 1e-15
    )

    warnings = []
    if final_gap > args.max_gap:
        warnings.append \
        (
            f"final generalization gap is {final_gap:.1%} "
            f"(limit {args.max_gap:.1%})"
        )
    if degradation > args.max_degradation:
        warnings.append \
        (
            f"validation loss degraded {degradation:.1%} after its best epoch "
            f"(limit {args.max_degradation:.1%})"
        )

    print(f"History:         {history_path}")
    print(f"Metric:          {args.metric}")
    print(f"Best validation: {best_validation:.6e} at epoch {epochs[best_index]}")
    print(f"Final train:     {train[-1]:.6e}")
    print(f"Final validation:{validation[-1]:.6e}")
    print(f"Final gap:       {final_gap:.1%}")
    print(f"Late degradation:{degradation:.1%}")
    score_random_reconstructions(args, run_dir)
    if warnings:
        print("RESULT: possible overfitting")
        for warning in warnings:
            print(f"  - {warning}")
        return 1

    print("RESULT: no clear overfitting signal")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
