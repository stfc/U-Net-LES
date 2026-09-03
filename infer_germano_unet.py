#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, os, shutil
from pathlib import Path
import matplotlib.pyplot as plt
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
        "--inverse-method",
        choices=("checkpoint",) + INVERSE_METHODS,
        default="checkpoint",
        help="Reconstruction operator (default: checkpoint, falling back to exact)",
    )
    p.add_argument("--van-cittert-iterations", type=int)
    p.add_argument("--van-cittert-relaxation", type=float)
    return p.parse_args()


def recreate_output_directory(path):
    """Remove a previous inference directory and create an empty one."""
    resolved = path.resolve()
    if resolved in (Path.cwd().resolve(), Path(resolved.anchor)):
        raise ValueError(f"Refusing to clear unsafe output directory: {path}")
    if path.is_symlink():
        raise ValueError(f"Refusing to clear symlinked output directory: {path}")
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"Output path exists but is not a directory: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True)


def main():
    a = parse_args()
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
    training_parameters = c.get("training_parameters", {})
    inverse_method = (
        c.get("inverse_method", training_parameters.get("inverse_method", "exact"))
        if a.inverse_method == "checkpoint"
        else a.inverse_method
    )
    van_cittert_iterations = (
        c.get(
            "van_cittert_iterations",
            training_parameters.get("van_cittert_iterations", 1),
        )
        if a.van_cittert_iterations is None
        else a.van_cittert_iterations
    )
    van_cittert_relaxation = (
        c.get(
            "van_cittert_relaxation",
            training_parameters.get("van_cittert_relaxation", 1.0),
        )
        if a.van_cittert_relaxation is None
        else a.van_cittert_relaxation
    )
    if van_cittert_iterations < 0:
        raise ValueError("--van-cittert-iterations must be non-negative.")
    if not 0 < van_cittert_relaxation <= 1:
        raise ValueError("--van-cittert-relaxation must be in (0, 1].")
    base_channels = int(c["base_channels"])
    model_output_channels = int(c["model_output_channels"])
    vorticity_mode = c["vorticity_mode"]
    vorticity_from_potential = vorticity_mode == "laplacian_of_potential"
    dns_vorticity_mode = c.get(
        "dns_vorticity_mode",
        "derived_from_potential"
        if vorticity_from_potential
        else "independent_channel",
    )
    cutoff_fraction = optional_environment_float(
        "PBOUT_UNET_INVERSE_CUTOFF_FRACTION",
        c.get("inverse_cutoff_fraction"),
    )
    cutoff_order = float(
        os.environ.get(
            "PBOUT_UNET_INVERSE_CUTOFF_ORDER",
            c.get("inverse_cutoff_order", 8.0),
        )
    )
    post_cutoff_fraction = optional_environment_float(
        "PBOUT_UNET_POST_CUTOFF_FRACTION",
        c.get("post_cutoff_fraction"),
    )
    post_cutoff_order = float(
        os.environ.get(
            "PBOUT_UNET_POST_CUTOFF_ORDER",
            c.get("post_cutoff_order", 8.0),
        )
    )
    post_filter_type = os.environ.get(
        "PBOUT_UNET_POST_FILTER",
        c.get(
            "post_filter",
            c.get("training_parameters", {}).get("post_filter", "high_order"),
        ),
    ).lower().replace("-", "_")
    if post_filter_type == "smooth_exponential_cutoff":
        post_filter_type = "high_order"
    if post_filter_type not in POST_FILTER_TYPES:
        raise ValueError(
            "PBOUT_UNET_POST_FILTER must be one of "
            f"{', '.join(POST_FILTER_TYPES)}."
        )
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
        "inverse_method": inverse_method,
        "van_cittert_iterations": van_cittert_iterations,
        "van_cittert_relaxation": van_cittert_relaxation,
        "inverse_cutoff_fraction": cutoff_fraction,
        "inverse_cutoff_order": cutoff_order,
        "post_cutoff_fraction": post_cutoff_fraction,
        "post_cutoff_order": post_cutoff_order,
        "post_filter": post_filter_type,
        "cutoff_application": {
            "inverse": "inverse_input_before_inverse_germano",
            "post": "reconstructed_dns_after_inverse_germano",
        },
        "vorticity_mode": vorticity_mode,
        "dns_vorticity_mode": dns_vorticity_mode,
        "vorticity_operator": c.get("vorticity_operator"),
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
    output_mean = mean[:, :model_output_channels]
    output_std = std[:, :model_output_channels]
    files = [a.input] if a.input.is_file() else recursive_npz_files(a.input)
    if not files:
        raise ValueError("No NPZ snapshots found.")
    recreate_output_directory(a.output_dir)
    effective_inference["snapshots_processed"] = len(files)
    checkpoint_metadata_path, checkpoint_metadata = print_checkpoint_metadata(
        a.checkpoint, effective_inference
    )
    rows = []
    for i, p in enumerate(files, 1):
        dns = load_dns_snapshot(
            p, fields, fine_n, lx, ly, dtype=numpy_dtype
        )
        coarse, fine = filter_snapshot(
            dns, alpha, lx, ly, coarse_n, dtype=numpy_dtype
        )
        x = (
            torch.from_numpy(((coarse - mean_np[0]) / std_np[0])[None])
            .to(device=device, dtype=torch_dtype)
        )
        with torch.no_grad():
            if a.inverse_input == "unet":
                pf = model(x) * output_std + output_mean
            else:
                pf = torch.from_numpy(
                    fine[None, :model_output_channels]
                ).to(device)
            inverse_input = spectral_cutoff_torch(
                pf,
                lx,
                ly,
                cutoff_fraction,
                cutoff_order,
            )
            pd_unconstrained = reconstruct_germano_torch(
                inverse_input,
                alpha,
                lx,
                ly,
                inverse_method,
                van_cittert_iterations,
                van_cittert_relaxation,
            )
            pd_unconstrained = post_filter_torch(
                pd_unconstrained,
                lx,
                ly,
                post_filter_type,
                post_cutoff_fraction,
                post_cutoff_order,
            )
            if vorticity_from_potential:
                vorticity = periodic_laplacian_2nd_torch(
                    pd_unconstrained[:, 1:2], lx, ly
                )
                pd_unconstrained = torch.cat(
                    (pd_unconstrained, vorticity), dim=1
                )
            pd = enforce_zero_spatial_mean(pd_unconstrained)
            dns_reference = torch.from_numpy(dns[None]).to(
                device=device, dtype=torch_dtype
            )
            pd = restore_channel_spatial_mean(pd, dns_reference, 2)
            diagnostic_filtered = pf
            diagnostic_inverse_input = inverse_input
            if vorticity_from_potential:
                filtered_vorticity = periodic_laplacian_2nd_torch(
                    pf[:, 1:2], lx, ly
                )
                filtered_vorticity = restore_channel_spatial_mean(
                    torch.cat((pf, filtered_vorticity), dim=1),
                    dns_reference,
                    2,
                )[:, 2:3]
                diagnostic_filtered = torch.cat(
                    (pf, filtered_vorticity), dim=1
                )
                filtered_vorticity = periodic_laplacian_2nd_torch(
                    inverse_input[:, 1:2], lx, ly
                )
                filtered_vorticity = restore_channel_spatial_mean(
                    torch.cat((inverse_input, filtered_vorticity), dim=1),
                    dns_reference,
                    2,
                )[:, 2:3]
                diagnostic_inverse_input = torch.cat(
                    (inverse_input, filtered_vorticity), dim=1
                )
        pf_np = diagnostic_filtered[0].cpu().numpy()
        inverse_input_np = diagnostic_inverse_input[0].cpu().numpy()
        pd_np = pd[0].cpu().numpy()
        out = a.output_dir / p.stem
        out.mkdir(parents=True, exist_ok=True)
        save_dns_centerline_plots(p.name, fields, dns, pd_np, out)
        reconstruction = {
            "dns_256": dns,
            "filtered_256": fine,
            "filtered_32": coarse,
            "inverse_input_256": inverse_input_np,
            "inverse_input_mode": np.asarray(a.inverse_input),
            "dns_vorticity_mode": np.asarray(dns_vorticity_mode),
            "vorticity_mode": np.asarray(vorticity_mode),
            "reconstructed_dns_256": pd_np,
            "field_names": np.asarray(fields),
            "source_file": np.asarray(str(p)),
        }
        if a.inverse_input == "unet":
            reconstruction["unet_filtered_256"] = pf_np
        np.savez_compressed(out / "reconstruction.npz", **reconstruction)
        for ch, name in enumerate(fields):
            cmap = {"U": "Blues", "V": "Reds_r", "P": "hot"}[name]
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
                f"{label}: Filtered {fine_n} (filtered DNS)",
                out / f"{name}_Filtered_{fine_n}.png",
                cmap,
            )
            save_image(
                coarse[ch],
                f"{label}: Filtered {coarse_n} (downsampled filtered image)",
                out / f"{name}_Filtered_{coarse_n}.png",
                cmap,
            )
            save_image(
                inverse_input_np[ch],
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
            inverse_input_np,
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
