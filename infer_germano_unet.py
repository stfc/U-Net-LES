#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, os
from pathlib import Path
import numpy as np, torch
from germano_common import *
from visualization_common import (
    save_arakawa_comparison,
    save_field_spectrum_comparison,
)


def inverse_input_label(mode, cutoff_fraction=None):
    if mode == "unet":
        if cutoff_fraction is not None:
            return "Cutoff-filtered U-Net outcome (inverse input)"
        return "U-Net filtered (U-Net outcome)"
    return "Filtered DNS (inverse input)"


def inverse_input_filename(mode):
    if mode == "unet":
        return "U-Net_filtered"
    return "Filtered_DNS_inverse_input"


def optional_environment_float(name, default):
    """Read an optional pBOUT float, accepting its cutoff-disable values."""
    value = os.environ.get(name)
    if value is None:
        return default
    if value.lower() in {"none", "off", "disable", "disabled"}:
        return None
    return float(value)


def environment_bool(name, default=True):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off"}


def print_checkpoint_metadata(checkpoint_path, effective_inference):
    """Load/print arbitrary training metadata and effective runtime overrides."""
    candidates = (
        checkpoint_path.parent / "metadata.json",
        checkpoint_path.parent.parent / "metadata.json",
    )
    metadata_path = next((path for path in candidates if path.is_file()), None)
    if metadata_path is None:
        print(
            "Checkpoint metadata: metadata.json was not found beside the "
            "checkpoint directory."
        )
        for name, value in effective_inference.items():
            print(f"effective_inference.{name}: {json.dumps(value)}")
        return None, {}

    metadata = json.loads(metadata_path.read_text())
    print(f"Checkpoint metadata ({metadata_path}):")

    def print_values(value, prefix=""):
        if isinstance(value, dict):
            for name, nested_value in value.items():
                if not prefix and name in {
                    "tensorboard_dir",
                    "test_metrics",
                    "training_parameters",
                    "training_parameters_file",
                }:
                    continue
                nested_prefix = f"{prefix}.{name}" if prefix else name
                print_values(nested_value, nested_prefix)
        else:
            print(f"{prefix}: {json.dumps(value)}")

    print_values(metadata)
    print_values(effective_inference, "effective_inference")
    return metadata_path, metadata


def save_dns_centerline_plots(source, fields, dns, reconstructed, output_dir):
    """Plot DNS and reconstructed values along y = N_DNS // 2."""
    center_y = dns.shape[-1] // 2
    x_index = np.arange(dns.shape[-2])
    for channel, name in enumerate(fields):
        fig, ax = plt.subplots(figsize=(8.0, 4.8))
        ax.plot(
            x_index,
            dns[channel, :, center_y],
            linewidth=1.8,
            label="DNS",
        )
        ax.plot(
            x_index,
            reconstructed[channel, :, center_y],
            "--",
            linewidth=1.8,
            label="Reconstructed DNS",
        )
        ax.set_title(
            f"{HW_FIELD_LABELS[name]} centerline comparison, "
            f"y={center_y}: {source}"
        )
        ax.set_xlabel("x index")
        ax.set_ylabel(HW_FIELD_LABELS[name])
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            output_dir / f"{name}_DNS_centerline_comparison.png", dpi=180
        )
        plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument(
        "--output-dir", type=Path, default=Path("germano_inference_output")
    )
    p.add_argument(
        "--inverse-input",
        choices=("unet", "filtered-dns"),
        default="unet",
        help="Field passed to the inverse Germano filter (default: unet)",
    )
    p.add_argument(
        "--precision",
        choices=("checkpoint", "float32", "float64"),
        default="checkpoint",
        help="U-Net and physics precision (default: use checkpoint precision)",
    )
    p.add_argument(
        "--laplacian-order",
        choices=("checkpoint", "2", "4"),
        default="checkpoint",
        help=(
            "Vorticity Laplacian order; use 2 or 4 to override the checkpoint "
            "(default: checkpoint)"
        ),
    )
    return p.parse_args()


def main():
    a = parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    c = load_checkpoint(a.checkpoint, device)
    validate_hw_checkpoint(c)
    precision = (
        c.get("precision", "float32")
        if a.precision == "checkpoint"
        else a.precision
    )
    torch_dtype = getattr(torch, precision)
    numpy_dtype = getattr(np, precision)
    fields = c["field_names"]
    fine_n = int(c["fine_n"])
    coarse_n = int(c["coarse_n"])
    lx = float(c["lx"])
    ly = float(c["ly"])
    alpha = float(c["alpha"])
    base_channels = int(c["base_channels"])
    vorticity_mode = c.get("vorticity_mode", "potential_laplacian")
    vorticity_operator = c.get(
        "vorticity_operator",
        c.get("training_parameters", {}).get(
            "vorticity_operator", "fourth_order_periodic_laplacian"
        ),
    )
    checkpoint_laplacian_order = int(
        c.get(
            "laplacian_order",
            2 if vorticity_operator == "second_order_periodic_laplacian" else 4,
        )
    )
    laplacian_order = (
        checkpoint_laplacian_order
        if a.laplacian_order == "checkpoint"
        else int(a.laplacian_order)
    )
    learn_vorticity = vorticity_mode == "learned_channel"
    model_output_channels = int(
        c.get("model_output_channels", 3 if learn_vorticity else 2)
    )
    expected_output_channels = 3 if learn_vorticity else 2
    if model_output_channels != expected_output_channels:
        raise ValueError(
            f"Checkpoint vorticity mode {vorticity_mode!r} requires "
            f"{expected_output_channels} model outputs, but metadata specifies "
            f"{model_output_channels}."
        )
    cutoff_fraction = optional_environment_float(
        "PBOUT_UNET_INVERSE_CUTOFF_FRACTION",
        c.get("inverse_cutoff_fraction", 0.75),
    )
    cutoff_order = float(
        os.environ.get(
            "PBOUT_UNET_INVERSE_CUTOFF_ORDER",
            c.get("inverse_cutoff_order", 8.0),
        )
    )
    post_cutoff_fraction = optional_environment_float(
        "PBOUT_UNET_POST_CUTOFF_FRACTION", 0.5
    )
    post_cutoff_order = float(
        os.environ.get("PBOUT_UNET_POST_CUTOFF_ORDER", 8.0)
    )
    requested_vorticity_from_potential = environment_bool(
        "PBOUT_UNET_VORTICITY_FROM_POTENTIAL", True
    )
    vorticity_from_potential = requested_vorticity_from_potential
    if not learn_vorticity and not requested_vorticity_from_potential:
        print(
            "PBOUT_UNET_VORTICITY_FROM_POTENTIAL=0 cannot provide an "
            "independently learned W with this potential-laplacian checkpoint; "
            "W will still be reconstructed from V."
        )
        vorticity_from_potential = True
    if cutoff_fraction is not None and not 0 < cutoff_fraction <= 1:
        raise ValueError(
            "PBOUT_UNET_INVERSE_CUTOFF_FRACTION must be in (0, 1]."
        )
    if post_cutoff_fraction is not None and not 0 < post_cutoff_fraction <= 1:
        raise ValueError(
            "PBOUT_UNET_POST_CUTOFF_FRACTION must be in (0, 1]."
        )
    if cutoff_order <= 0:
        raise ValueError("PBOUT_UNET_INVERSE_CUTOFF_ORDER must be positive.")
    if post_cutoff_order <= 0:
        raise ValueError("PBOUT_UNET_POST_CUTOFF_ORDER must be positive.")
    normalization_enabled = c.get(
        "normalize",
        c.get("training_parameters", {}).get("normalize", True),
    )
    effective_inference = {
        "checkpoint": str(a.checkpoint),
        "checkpoint_epoch": c.get("epoch"),
        "input": str(a.input),
        "device": str(device),
        "precision": precision,
        "physics_dtype": precision,
        "normalization_enabled": normalization_enabled,
        "inverse_input": a.inverse_input,
        "inverse_cutoff_fraction": cutoff_fraction,
        "inverse_cutoff_order": cutoff_order,
        "post_cutoff_fraction": post_cutoff_fraction,
        "post_cutoff_order": post_cutoff_order,
        "cutoff_application": {
            "inverse": "inverse_input_before_inverse_germano",
            "post": "reconstructed_dns_after_inverse_germano",
        },
        "requested_vorticity_from_potential": (
            requested_vorticity_from_potential
        ),
        "vorticity_from_potential": vorticity_from_potential,
        "vorticity_mode": vorticity_mode,
        "vorticity_operator": vorticity_operator,
        "checkpoint_laplacian_order": checkpoint_laplacian_order,
        "laplacian_order": laplacian_order,
    }
    model = ResidualUNetSR(
        len(fields),
        base_channels,
        fine_n // coarse_n,
        model_output_channels,
    ).to(device=device, dtype=torch_dtype)
    model.load_state_dict(c["model"])
    model.eval()
    mean_np = np.asarray(c["mean"], dtype=numpy_dtype)
    std_np = np.asarray(c["std"], dtype=numpy_dtype)
    mean = torch.from_numpy(mean_np).to(device)
    std = torch.from_numpy(std_np).to(device)
    files = [a.input] if a.input.is_file() else recursive_npz_files(a.input)
    if not files:
        raise ValueError("No NPZ snapshots found.")
    effective_inference["snapshots_processed"] = len(files)
    checkpoint_metadata_path, checkpoint_metadata = print_checkpoint_metadata(
        a.checkpoint, effective_inference
    )
    rows = []
    for i, p in enumerate(files, 1):
        dns = load_dns_snapshot(
            p, fields, fine_n, lx, ly, dtype=numpy_dtype,
            laplacian_order=laplacian_order
        )
        coarse, fine = filter_snapshot(
            dns, alpha, lx, ly, coarse_n, dtype=numpy_dtype,
            laplacian_order=laplacian_order
        )
        x = (
            torch.from_numpy(((coarse - mean_np[0]) / std_np[0])[None])
            .to(device=device, dtype=torch_dtype)
        )
        with torch.no_grad():
            if a.inverse_input == "unet":
                predicted = model(x)
                if learn_vorticity:
                    pf = predicted * std + mean
                else:
                    predicted_uv = (
                        predicted * std[:, :2] + mean[:, :2]
                    )
                    pf = hw_fields_torch(
                        predicted_uv[:, :1], predicted_uv[:, 1:2], lx, ly,
                        laplacian_order=laplacian_order
                    )
            else:
                pf = torch.from_numpy(fine[None]).to(device)
            if vorticity_from_potential:
                pf = enforce_vorticity_from_potential_torch(
                    pf, lx, ly, laplacian_order
                )
            pf = spectral_cutoff_torch(
                pf,
                lx,
                ly,
                cutoff_fraction,
                cutoff_order,
            )
            if vorticity_from_potential:
                pf = enforce_vorticity_from_potential_torch(
                    pf, lx, ly, laplacian_order
                )
            pd_unconstrained = inverse_germano_torch(
                pf,
                alpha,
                lx,
                ly,
            )
            if vorticity_from_potential:
                pd_unconstrained = enforce_vorticity_from_potential_torch(
                    pd_unconstrained, lx, ly, laplacian_order
                )
            pd_unconstrained = spectral_cutoff_torch(
                pd_unconstrained,
                lx,
                ly,
                post_cutoff_fraction,
                post_cutoff_order,
            )
            if vorticity_from_potential:
                pd = enforce_vorticity_from_potential_torch(
                    pd_unconstrained, lx, ly, laplacian_order
                )
            else:
                pd = pd_unconstrained
        pf_np, pd_np = pf[0].cpu().numpy(), pd[0].cpu().numpy()
        out = a.output_dir / p.stem
        out.mkdir(parents=True, exist_ok=True)
        save_dns_centerline_plots(p.name, fields, dns, pd_np, out)
        reconstruction = {
            "dns_256": dns,
            "filtered_256": fine,
            "filtered_32": coarse,
            "inverse_input_256": pf_np,
            "inverse_input_mode": np.asarray(a.inverse_input),
            "reconstructed_dns_256": pd_np,
            "field_names": np.asarray(fields),
            "source_file": np.asarray(str(p)),
        }
        if a.inverse_input == "unet":
            reconstruction["unet_filtered_256"] = pf_np
        np.savez_compressed(out / "reconstruction.npz", **reconstruction)
        for ch, name in enumerate(fields):
            cmap = {"U": "Blues", "V": "Reds_r", "W": "hot"}[name]
            label = HW_FIELD_LABELS[name]
            de = pd_np[ch] - dns[ch]
            fe = pf_np[ch] - fine[ch]
            rows.append(
                {
                    "snapshot": p.name,
                    "field": name,
                    "filtered_relative_l2": float(
                        np.linalg.norm(fe.ravel())
                        / np.linalg.norm(fine[ch].ravel())
                    ),
                    "dns_relative_l2": float(
                        np.linalg.norm(de.ravel())
                        / np.linalg.norm(dns[ch].ravel())
                    ),
                }
            )
            save_image(
                dns[ch],
                f"{label}: DNS (original DNS)",
                out / f"{name}_DNS.png",
                cmap,
            )
            save_image(
                fine[ch],
                f"{label}: Filtered 256 (filtered DNS)",
                out / f"{name}_Filtered_256.png",
                cmap,
            )
            save_image(
                coarse[ch],
                f"{label}: Filtered 32 (downsampled filtered image)",
                out / f"{name}_Filtered_32.png",
                cmap,
            )
            save_image(
                pf_np[ch],
                f"{label}: {inverse_input_label(a.inverse_input, cutoff_fraction)}",
                out / f"{name}_{inverse_input_filename(a.inverse_input)}.png",
                cmap,
            )
            if a.inverse_input == "unet":
                save_image(
                    fine[ch] - pf_np[ch],
                    f"{label}: Filtered DNS minus U-Net filtered",
                    out / f"{name}_Filtered_DNS_minus_U-Net_filtered.png",
                    "coolwarm",
                    symmetric=True,
                )
            save_image(
                pd_np[ch],
                f"{label}: Reconstructed DNS",
                out / f"{name}_Reconstructed_DNS.png",
                cmap,
            )
            save_image(
                de,
                f"{label}: DNS reconstruction error",
                out / f"{name}_DNS_reconstruction_error.png",
                cmap,
            )
        save_spectrum_plot(
            dns,
            fine,
            coarse,
            pf_np,
            pd_np,
            coarse_n,
            lx,
            ly,
            out / "HW_energy_spectrum_comparison.png",
            inverse_input_label(a.inverse_input, cutoff_fraction),
        )
        save_field_spectrum_comparison(
            pd_np,
            dns,
            fields,
            lx,
            ly,
            out / "reconstructed_DNS_spectrum_comparison.png",
            post_cutoff_fraction,
        )
        save_arakawa_comparison(
            pd_np,
            dns,
            lx,
            ly,
            out,
            cutoff_fraction=post_cutoff_fraction,
        )
        print(f"Processed {i}/{len(files)}: {p.name}")
    with (a.output_dir / "inference_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    effective_inference["checkpoint_metadata_file"] = (
        str(checkpoint_metadata_path)
        if checkpoint_metadata_path is not None
        else None
    )
    metadata = {
        "checkpoint_metadata": checkpoint_metadata,
        "effective_inference": effective_inference,
    }
    (a.output_dir / "inference_metadata.json").write_text(
        json.dumps(metadata, indent=2)
    )


if __name__ == "__main__":
    main()
