#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np, torch
from germano_common import *


def inverse_input_label(mode):
    if mode == "unet":
        return "U-Net filtered (U-Net outcome)"
    return "Filtered DNS (inverse input)"


def inverse_input_filename(mode):
    if mode == "unet":
        return "U-Net_filtered"
    return "Filtered_DNS_inverse_input"


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
        "--inverse-cutoff-fraction",
        type=float,
        help="Override checkpoint cutoff as a fraction of grid Nyquist",
    )
    p.add_argument(
        "--inverse-cutoff-order",
        type=float,
        help="Override checkpoint exponential cutoff order",
    )
    p.add_argument(
        "--disable-inverse-cutoff",
        action="store_true",
        help="Use the exact inverse Germano operator",
    )
    return p.parse_args()


def main():
    a = parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    c = load_checkpoint(a.checkpoint, device)
    validate_hw_checkpoint(c)
    fields = c["field_names"]
    fine_n = int(c["fine_n"])
    coarse_n = int(c["coarse_n"])
    lx = float(c["lx"])
    ly = float(c["ly"])
    alpha = float(c["alpha"])
    base_channels = int(c["base_channels"])
    vorticity_mode = c.get("vorticity_mode", "potential_laplacian")
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
    cutoff_fraction = a.inverse_cutoff_fraction
    if cutoff_fraction is None:
        cutoff_fraction = c.get("inverse_cutoff_fraction", 0.75)
    cutoff_order = a.inverse_cutoff_order
    if cutoff_order is None:
        cutoff_order = c.get("inverse_cutoff_order", 8.0)
    if cutoff_fraction is not None and not 0 < cutoff_fraction <= 1:
        raise ValueError("inverse-cutoff-fraction must be in (0, 1].")
    if cutoff_order <= 0:
        raise ValueError("inverse-cutoff-order must be positive.")
    if a.disable_inverse_cutoff or a.inverse_input == "filtered-dns":
        cutoff_fraction = None
    model = ResidualUNetSR(
        len(fields),
        base_channels,
        fine_n // coarse_n,
        model_output_channels,
    ).to(device)
    model.load_state_dict(c["model"])
    model.eval()
    mean_np = np.asarray(c["mean"], np.float32)
    std_np = np.asarray(c["std"], np.float32)
    mean = torch.from_numpy(mean_np).to(device)
    std = torch.from_numpy(std_np).to(device)
    files = [a.input] if a.input.is_file() else recursive_npz_files(a.input)
    if not files:
        raise ValueError("No NPZ snapshots found.")
    rows = []
    for i, p in enumerate(files, 1):
        dns = load_dns_snapshot(
            p, fields, fine_n, lx, ly, dtype=np.float64
        )
        coarse, fine = filter_snapshot(
            dns, alpha, lx, ly, coarse_n, dtype=np.float64
        )
        x = (
            torch.from_numpy(((coarse - mean_np[0]) / std_np[0])[None])
            .float()
            .to(device)
        )
        with torch.no_grad():
            if a.inverse_input == "unet":
                predicted = model(x).double()
                if learn_vorticity:
                    pf = predicted * std.double() + mean.double()
                else:
                    predicted_uv = (
                        predicted * std[:, :2].double()
                        + mean[:, :2].double()
                    )
                    pf = hw_fields_torch(
                        predicted_uv[:, :1], predicted_uv[:, 1:2], lx, ly
                    )
            else:
                pf = torch.from_numpy(fine[None]).to(device)
            pd_unconstrained = inverse_germano_torch(
                pf,
                alpha,
                lx,
                ly,
                cutoff_fraction,
                cutoff_order,
            )
            if learn_vorticity:
                pd = pd_unconstrained
            else:
                pd = hw_fields_torch(
                    pd_unconstrained[:, :1],
                    pd_unconstrained[:, 1:2],
                    lx,
                    ly,
                )
        pf_np, pd_np = pf[0].cpu().numpy(), pd[0].cpu().numpy()
        out = a.output_dir / p.stem
        out.mkdir(parents=True, exist_ok=True)
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
                f"{label}: {inverse_input_label(a.inverse_input)}",
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
                name,
                dns[ch],
                fine[ch],
                coarse[ch],
                pf_np[ch],
                pd_np[ch],
                coarse_n,
                lx,
                ly,
                out / f"{name}_spectrum_comparison.png",
                inverse_input_label(a.inverse_input),
            )
        print(f"Processed {i}/{len(files)}: {p.name}")
    with (a.output_dir / "inference_metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    (a.output_dir / "inference_metadata.json").write_text(
        json.dumps(
            {
                "checkpoint": str(a.checkpoint),
                "checkpoint_epoch": int(c["epoch"]),
                "input": str(a.input),
                "snapshots_processed": len(files),
                "device": str(device),
                "inverse_input": a.inverse_input,
                "physics_dtype": "float64",
                "inverse_cutoff_fraction": cutoff_fraction,
                "inverse_cutoff_order": cutoff_order,
                "vorticity_mode": vorticity_mode,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
