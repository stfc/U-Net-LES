#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np, torch
from germano_common import *


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument(
        "--output-dir", type=Path, default=Path("germano_inference_output")
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
    model = ResidualUNetSR(len(fields), base_channels, fine_n // coarse_n).to(
        device
    )
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
        dns = load_dns_snapshot(p, fields, fine_n, lx, ly)
        coarse, fine = filter_snapshot(dns, alpha, lx, ly, coarse_n)
        x = (
            torch.from_numpy(((coarse - mean_np[0]) / std_np[0])[None])
            .float()
            .to(device)
        )
        with torch.no_grad():
            predicted_uv = model(x) * std[:, :2] + mean[:, :2]
            pf = hw_fields_torch(
                predicted_uv[:, :1], predicted_uv[:, 1:2], lx, ly
            )
            pd_unconstrained = inverse_germano_torch(pf, alpha, lx, ly)
            pd = hw_fields_torch(
                pd_unconstrained[:, :1],
                pd_unconstrained[:, 1:2],
                lx,
                ly,
            )
        pf_np, pd_np = pf[0].cpu().numpy(), pd[0].cpu().numpy()
        out = a.output_dir / p.stem
        out.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            out / "reconstruction.npz",
            dns_256=dns,
            filtered_256=fine,
            filtered_32=coarse,
            unet_filtered_256=pf_np,
            reconstructed_dns_256=pd_np,
            field_names=np.asarray(fields),
            source_file=np.asarray(str(p)),
        )
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
                coarse[ch],
                f"{label}: filtered field ({coarse_n} x {coarse_n})",
                out / f"{name}_filtered_32.png",
                cmap,
            )
            save_image(
                pf_np[ch],
                f"{label}: U-Net reconstructed filtered field",
                out / f"{name}_unet_filtered_256.png",
                cmap,
            )
            save_image(
                pd_np[ch],
                f"{label}: reconstructed DNS",
                out / f"{name}_reconstructed_dns_256.png",
                cmap,
            )
            save_image(
                de,
                f"{label}: DNS reconstruction error",
                out / f"{name}_dns_error.png",
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
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
