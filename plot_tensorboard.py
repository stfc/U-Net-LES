#!/usr/bin/env python3
"""Export TensorBoard scalar histories as PNG figures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)


PLOT_GROUPS = {
    "losses": ("total", "filtered", "dns"),
    "relative_l2": ("filtered_relative_l2", "dns_relative_l2"),
    "dns_field_losses": ("dns_U", "dns_V", "dns_P"),
}

PANEL_LAYOUT = [
    ["definitions", "total", "learning_rate", "."],
    ["filtered", "dns", ".", "."],
    ["filtered_relative_l2", "dns_relative_l2", ".", "."],
    [
        "dns_U",
        "dns_V",
        "dns_P",
        ".",
    ],
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot scalar metrics stored in TensorBoard event files."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help=(
            "Training output directory or its tensorboard subdirectory"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="PNG destination (default: RUN_DIR/tensorboard_plots)",
    )
    parser.add_argument(
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


def load_loss_weights(run_dir: Path):
    run_root = run_dir.parent if run_dir.name == "tensorboard" else run_dir
    parameters_path = run_root / "training_parameters.json"
    if not parameters_path.is_file():
        return None
    parameters = json.loads(parameters_path.read_text())
    return parameters.get("lambda_dns")


def format_weight(value):
    return "unknown" if value is None else f"{value:g}"


def plot_all_metrics(
    accumulator,
    output_path,
    log_scale,
    lambda_dns=None,
):
    available = set(accumulator.Tags().get("scalars", ()))
    fig, axes = plt.subplot_mosaic(
        PANEL_LAYOUT,
        figsize=(20.8, 17.6),
        gridspec_kw={"height_ratios": (1, 1, 1, 1)},
    )
    plotted = False
    for metric in (
        "total",
        "learning_rate",
        "filtered",
        "dns",
        "filtered_relative_l2",
        "dns_relative_l2",
        "dns_U",
        "dns_V",
        "dns_P",
    ):
        ax = axes[metric]
        if metric == "learning_rate":
            steps, values = scalar_series(
                accumulator, "optimizer/learning_rate"
            ) if "optimizer/learning_rate" in available else ([], [])
            if steps:
                ax.plot(steps, values, color="tab:green")
                plotted = True
            ax.set_title("Learning Rate")
        else:
            for split, color in (
                ("train", "tab:blue"),
                ("validation", "tab:orange"),
            ):
                tag = f"metrics/{split}/{metric}"
                if tag not in available:
                    continue
                steps, values = scalar_series(accumulator, tag)
                ax.plot(steps, values, color=color, label=split)
                plotted = True
            ax.set_title(metric.replace("_", " ").title())
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Value")
        if log_scale:
            ax.set_yscale("log")
        ax.grid(True, which="both", alpha=0.25)
        if ax.lines:
            if metric != "learning_rate":
                ax.legend()
        else:
            ax.text(
                0.5,
                0.5,
                "No scalar data",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
    definition_ax = axes["definitions"]
    definition_ax.set_axis_off()
    definitions = (
        "Definitions\n"
        r"$\mathcal{L}_{total}"
        r"=\mathcal{L}_{filtered}"
        r"+\lambda_{DNS}\mathcal{L}_{DNS}$"
        "\n"
        r"$L_{2,\mathrm{relative}}"
        r"=\dfrac{\|\hat{q}-q\|_2}{\|q\|_2}$"
        "\n"
        r"$\lambda_{filtered}=1,\quad"
        rf"\lambda_{{DNS}}={format_weight(lambda_dns)}$"
        "\n"
        "Total: weighted loss; component curves: unweighted losses\n"
        r"$\hat{q}$: prediction"
        "        "
        r"$q$: target"
    )
    definition_ax.text(
        0.5,
        0.5,
        definitions,
        ha="center",
        va="center",
        fontsize=11,
        linespacing=1.25,
        transform=definition_ax.transAxes,
        bbox={
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
        raise FileNotFoundError(
            f"No TensorBoard event files were found in {log_dir}"
        )

    output_dir = args.output_dir or args.run_dir / "tensorboard_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    accumulator = EventAccumulator(
        str(log_dir), size_guidance={"scalars": 0}
    )
    accumulator.Reload()
    lambda_dns = load_loss_weights(args.run_dir)

    output_path = output_dir / "tensorboard_summary.png"
    if not plot_all_metrics(
        accumulator,
        output_path,
        log_scale=not args.linear_scale,
        lambda_dns=lambda_dns,
    ):
        available = accumulator.Tags().get("scalars", ())
        raise RuntimeError(
            "No recognized training scalars were found. Available tags: "
            + ", ".join(available)
        )
    print(f"Read TensorBoard events from {log_dir}")
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
