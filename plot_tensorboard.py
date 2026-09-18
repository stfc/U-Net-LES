#!/usr/bin/env python3
"""Export TensorBoard scalar histories as PNG figures."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import \
(
    EventAccumulator,
)


PLOT_GROUPS = \
{
    "losses": ("total", "dns", "dns_P", "dns_mse", "dns_P_mse", "kl"),
    "relative_l2": ("filtered_relative_l2", "dns_relative_l2"),
    "dns_field_losses": ("dns_U", "dns_V"),
}

PANEL_LAYOUT = \
[
    ["definitions", "definitions", "total", "learning_rate"],
    ["dns", "dns_U", "dns_V", "dns_P"],
    ["dns_mse", "dns_mse", "dns_P_mse", "dns_P_mse"],
    [
        "kl", "kl",
        "filtered_relative_l2", "dns_relative_l2"
    ],
]

METRIC_TITLES = \
{
    "total": "Total Training Objective",
    "dns": "DNS Density/Potential L1",
    "dns_U": "DNS Density L1",
    "dns_V": "DNS Potential L1",
    "dns_P": "DNS Vorticity L1 (Normalized)",
    "dns_mse": "DNS Density/Potential MSE",
    "dns_P_mse": "DNS Vorticity MSE (Normalized)",
    "kl": "Mean Latent KL Divergence",
    "filtered_relative_l2": "Filtered Relative L2 (Diagnostic Only)",
    "dns_relative_l2": "DNS Density/Potential Relative L2 (Diagnostic Only)",
}


def parse_args():
    parser = argparse.ArgumentParser \
    (
        description="Plot scalar metrics stored in TensorBoard event files."
    )
    parser.add_argument \
    (
        "run_dir",
        type=Path,
        help= \
        (
            "Training output directory or its tensorboard subdirectory"
        ),
    )
    parser.add_argument \
    (
        "--output-dir",
        type=Path,
        help="PNG destination (default: RUN_DIR/tensorboard_plots)",
    )
    parser.add_argument \
    (
        "--linear-scale",
        action="store_true",
        help="Use a linear y-axis instead of logarithmic scaling",
    )
    return parser.parse_args()


def resolve_log_dir(run_dir: Path) -> Path:
    tensorboard_dir = run_dir / "tensorboard"
    return tensorboard_dir if tensorboard_dir.is_dir() else run_dir


def scalar_series(accumulator, tag):
    events = accumulator.Scalars(tag)
    return [event.step for event in events], [event.value for event in events]


def load_training_parameters(run_dir: Path):
    run_root = run_dir.parent if run_dir.name == "tensorboard" else run_dir
    parameters_path = run_root / "training_parameters.json"
    if not parameters_path.is_file():
        return None
    return json.loads(parameters_path.read_text())


def format_weight(value):
    return "unknown" if value is None else f"{value:g}"


def plot_all_metrics \
(
    accumulator,
    output_path,
    log_scale,
    training_parameters=None,
):
    training_parameters = training_parameters or {}
    available = set(accumulator.Tags().get("scalars", ()))
    fig, axes = plt.subplot_mosaic \
    (
        PANEL_LAYOUT,
        figsize=(20.8, 13.4),
    )
    plotted = False
    for metric in \
    (
        "total",
        "learning_rate",
        "dns",
        "kl",
        "filtered_relative_l2",
        "dns_relative_l2",
        "dns_U",
        "dns_V",
        "dns_P",
        "dns_mse",
        "dns_P_mse",
    ):
        ax = axes[metric]
        if metric == "learning_rate":
            steps, values = scalar_series \
            (
                accumulator, "optimizer/learning_rate"
            ) if "optimizer/learning_rate" in available else ([], [])
            if steps:
                ax.plot(steps, values, color="tab:green")
                plotted = True
            ax.set_title("Learning Rate")
        else:
            for split, color in \
            (
                ("train", "tab:blue"),
                ("validation", "tab:orange"),
            ):
                tag = f"metrics/{split}/{metric}"
                if tag not in available:
                    continue
                steps, values = scalar_series(accumulator, tag)
                ax.plot(steps, values, color=color, label=split)
                plotted = True
            ax.set_title \
            (
                METRIC_TITLES.get \
                (
                    metric, metric.replace("_", " ").title()
                )
            )
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        # Empty or nonpositive metrics (for example disabled KL) need a linear axis.
        has_positive_values = any \
        (
            math.isfinite(value) and value > 0
            for line in ax.lines
            for value in line.get_ydata()
        )
        if log_scale and has_positive_values:
            ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        if ax.lines:
            if metric != "learning_rate":
                ax.legend()
        else:
            ax.text \
            (
                0.5,
                0.5,
                "No scalar data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
    definition_ax = axes["definitions"]
    definition_ax.set_axis_off()
    vorticity_weight = training_parameters.get("lambda_vorticity")
    if vorticity_weight is None:
        vorticity_weight = training_parameters.get \
        (
            "effective_loss_weights", {}
        ).get("vorticity")
    kl_weight = training_parameters.get("effective_loss_weights", {}).get \
    (
        "kl", training_parameters.get("lambda_kl")
    )
    if training_parameters.get("variational") is False:
        kl_weight = 0.0
    normalization = \
    (
        "Errors in n and phi are scaled by their DNS training standard deviations."
        if training_parameters.get("normalize", True)
        else "Errors in n and phi use physical values (no normalization)."
    )
    definitions = \
    (
        "Active training objective\n"
        r"$\mathcal{L}_{total}=\mathcal{L}_{n,\phi}"
        r"+\lambda_{\omega}\mathcal{L}_{\omega}"
        r"+\lambda_{MSE}(\mathcal{M}_{n,\phi}+\lambda_\omega\mathcal{M}_\omega)"
        r"+\lambda_{KL}\mathcal{L}_{KL}$"
        "\n"
        r"$\mathcal{L}_{n,\phi}=\dfrac{1}{2}\left("
        r"\operatorname{mean}|\Delta n/\sigma_n|+"
        r"\operatorname{mean}|\Delta\phi/\sigma_\phi|\right)$"
        "\n"
        r"$\mathcal{L}_{\omega}="
        r"\operatorname{mean}|\Delta\omega/\sigma_\omega|,\quad"
        rf"\lambda_{{\omega}}={format_weight(vorticity_weight)}$"
        "\n"
        + f"\nMSE weight: {training_parameters.get('lambda_mse', 0.0):g}; "
        "M uses squared normalized residuals.\n"
        + normalization
        + "\nErrors compare U-Net reconstruction with DNS targets after inversion."
        + "\nVorticity uses its DNS training standard deviation."
        + f"\nMean latent KL weight: {format_weight(kl_weight)}. "
        "Gradients pass through Germano inversion.\n"
        r"$L_{2,\mathrm{relative}}="
        r"\dfrac{\|\hat{q}-q\|_2}{\|q\|_2}$ (diagnostic only)."
    )
    definition_ax.text \
    (
        0.5,
        0.5,
        definitions,
        ha="center",
        va="center",
        fontsize=10,
        linespacing=1.25,
        transform=definition_ax.transAxes,
        bbox= \
        {
            "boxstyle": "round,pad=0.6",
            "facecolor": "whitesmoke",
            "edgecolor": "0.35",
            "linewidth": 1.2,
        },
    )
    if not plotted:
        plt.close(fig)
        return False
    fig.suptitle("TensorBoard Training Summary", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return True


def main():
    args = parse_args()
    log_dir = resolve_log_dir(args.run_dir)
    if not log_dir.is_dir():
        raise FileNotFoundError(f"TensorBoard directory not found: {log_dir}")
    if not list(log_dir.glob("events.out.tfevents.*")):
        raise FileNotFoundError \
        (
            f"No TensorBoard event files were found in {log_dir}"
        )

    output_dir = args.output_dir or args.run_dir / "tensorboard_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    accumulator = EventAccumulator \
    (
        str(log_dir), size_guidance={"scalars": 0}
    )
    accumulator.Reload()
    training_parameters = load_training_parameters(args.run_dir) or {}

    output_path = output_dir / "tensorboard_summary.png"
    if not plot_all_metrics \
    (
        accumulator,
        output_path,
        log_scale=not args.linear_scale,
        training_parameters=training_parameters,
    ):
        available = accumulator.Tags().get("scalars", ())
        raise RuntimeError \
        (
            "No recognized training scalars were found. Available tags: "
            + ", ".join(available)
        )
    print(f"Read TensorBoard events from {log_dir}")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
