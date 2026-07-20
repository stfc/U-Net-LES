#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, random
from pathlib import Path
import numpy as np, torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from germano_common import *


def split_files(dataset, validation_fraction, test_fraction, split_mode, seed):
    td, vd, sd = dataset / "train", dataset / "validation", dataset / "test"
    if td.is_dir() and vd.is_dir() and sd.is_dir():
        return (
            recursive_npz_files(td),
            recursive_npz_files(vd),
            recursive_npz_files(sd),
        )
    files = recursive_npz_files(dataset)
    if len(files) < 3:
        raise ValueError("At least three NPZ snapshots are required.")
    if split_mode == "random":
        random.Random(seed).shuffle(files)
    nv = max(1, round(validation_fraction * len(files)))
    nt = max(1, round(test_fraction * len(files)))
    ntr = len(files) - nv - nt
    if ntr < 1:
        raise ValueError("The split leaves no training snapshots.")
    return files[:ntr], files[ntr : ntr + nv], files[ntr + nv :]


def run_epoch(
    model,
    loader,
    device,
    mean,
    std,
    alpha,
    lx,
    ly,
    lambda_dns,
    lambda_spectrum,
    optimizer=None,
):
    training = optimizer is not None
    model.train(training)
    totals = {
        k: 0.0
        for k in [
            "total",
            "filtered",
            "dns",
            "spectrum",
            "filtered_relative_l2",
            "dns_relative_l2",
        ]
    }
    count = 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for coarse, ft, dns in loader:
            coarse, ft, dns = (
                coarse.to(device, non_blocking=True),
                ft.to(device, non_blocking=True),
                dns.to(device, non_blocking=True),
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
            loss, vals, pf, pd = calculate_losses(
                model,
                coarse,
                ft,
                dns,
                mean,
                std,
                alpha,
                lx,
                ly,
                lambda_dns,
                lambda_spectrum,
            )
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            b = coarse.shape[0]
            count += b
            for k in ["total", "filtered", "dns", "spectrum"]:
                totals[k] += vals[k].detach().item() * b

            totals["filtered_relative_l2"] += (
                relative_l2(ft * std + mean, pf).detach().item() * b
            )
            totals["dns_relative_l2"] += (
                relative_l2(dns, pd).detach().item() * b
            )

    return {k: v / count for k, v in totals.items()}


def configure_tensorboard(writer):
    writer.add_custom_scalars(
        {
            "Losses": {
                name.title(): [
                    "Multiline",
                    [f"metrics/train/{name}", f"metrics/validation/{name}"],
                ]
                for name in ("total", "filtered", "dns", "spectrum")
            },
            "Relative L2": {
                name.replace("_", " ").title(): [
                    "Multiline",
                    [f"metrics/train/{name}", f"metrics/validation/{name}"],
                ]
                for name in ("filtered_relative_l2", "dns_relative_l2")
            },
        }
    )


def log_epoch_metrics(writer, epoch, train_metrics, validation_metrics, lr):
    for name, value in train_metrics.items():
        writer.add_scalar(f"metrics/train/{name}", value, epoch)
    for name, value in validation_metrics.items():
        writer.add_scalar(f"metrics/validation/{name}", value, epoch)
    writer.add_scalar("optimizer/learning_rate", lr, epoch)
    writer.flush()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument(
        "--output-dir", type=Path, default=Path("germano_training_output")
    )
    p.add_argument("--fields", nargs=3, default=list(HW_FIELD_NAMES))
    p.add_argument("--fine-n", type=int, default=256)
    p.add_argument("--coarse-n", type=int, default=32)
    p.add_argument("--lx", type=float, default=2 * np.pi)
    p.add_argument("--ly", type=float, default=2 * np.pi)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--alpha", type=float)
    g.add_argument("--alpha-factor", type=float, default=1.0)
    p.add_argument(
        "--split-mode", choices=["sequential", "random"], default="sequential"
    )
    p.add_argument("--validation-fraction", type=float, default=0.1)
    p.add_argument("--test-fraction", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--learning-rate", type=float, default=2e-4)
    p.add_argument("--base-channels", type=int, default=24)
    p.add_argument("--lambda-dns", type=float, default=0.05)
    p.add_argument("--lambda-spectrum", type=float, default=0.02)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--resume", type=Path)
    p.add_argument(
        "--tensorboard-dir",
        type=Path,
        help="TensorBoard log directory (default: OUTPUT_DIR/tensorboard)",
    )
    return p.parse_args()


def main():
    a = parse_args()
    set_seed(a.seed)
    if tuple(a.fields) != HW_FIELD_NAMES:
        raise ValueError(f"Hasegawa-Wakatani fields must be {HW_FIELD_NAMES}.")
    if a.fine_n != 8 * a.coarse_n:
        raise ValueError("Current model requires fine_n = 8 * coarse_n.")
    a.output_dir.mkdir(parents=True, exist_ok=True)
    ckdir = a.output_dir / "checkpoints"
    ckdir.mkdir(exist_ok=True)
    tr, va, te = split_files(
        a.dataset, a.validation_fraction, a.test_fraction, a.split_mode, a.seed
    )
    delta = a.lx / a.fine_n
    alpha = a.alpha if a.alpha is not None else a.alpha_factor * delta
    mean_np, std_np = calculate_training_statistics(
        tr, a.fields, a.fine_n, a.coarse_n, alpha, a.lx, a.ly
    )

    def ds(files):
        return GermanoSnapshotDataset(
            files,
            a.fields,
            a.fine_n,
            a.coarse_n,
            alpha,
            a.lx,
            a.ly,
            mean_np,
            std_np,
        )

    train_ds, val_ds, test_ds = ds(tr), ds(va), ds(te)
    kw = dict(
        batch_size=a.batch_size,
        num_workers=a.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=a.num_workers > 0,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **kw)
    val_loader = DataLoader(val_ds, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, shuffle=False, **kw)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True
    model = ResidualUNetSR(
        len(a.fields), a.base_channels, a.fine_n // a.coarse_n
    ).to(device)
    opt = torch.optim.AdamW(
        model.parameters(), lr=a.learning_rate, weight_decay=1e-6
    )
    mean = torch.from_numpy(mean_np).to(device)
    std = torch.from_numpy(std_np).to(device)
    start = 1
    best = float("inf")
    if a.resume:
        c = load_checkpoint(a.resume, device)
        validate_hw_checkpoint(c)
        model.load_state_dict(c["model"])
        opt.load_state_dict(c["optimizer"])
        start = int(c["epoch"]) + 1
        best = float(c["validation_loss"])
        print("Resumed from epoch", start - 1)
    tensorboard_dir = a.tensorboard_dir or a.output_dir / "tensorboard"
    writer = SummaryWriter(log_dir=tensorboard_dir, purge_step=start)
    configure_tensorboard(writer)
    print("TensorBoard logs:", tensorboard_dir)
    history = []
    for epoch in range(start, a.epochs + 1):
        tm = run_epoch(
            model,
            train_loader,
            device,
            mean,
            std,
            alpha,
            a.lx,
            a.ly,
            a.lambda_dns,
            a.lambda_spectrum,
            opt,
        )
        vm = run_epoch(
            model,
            val_loader,
            device,
            mean,
            std,
            alpha,
            a.lx,
            a.ly,
            a.lambda_dns,
            a.lambda_spectrum,
        )
        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in tm.items()},
            **{f"validation_{k}": v for k, v in vm.items()},
        }
        history.append(row)
        log_epoch_metrics(
            writer,
            epoch,
            tm,
            vm,
            opt.param_groups[0]["lr"],
        )
        c = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "validation_loss": vm["total"],
            "alpha": alpha,
            "delta": delta,
            "mean": mean_np,
            "std": std_np,
            "field_names": list(a.fields),
            "physics_version": 1,
            "vorticity_operator": "fourth_order_periodic_laplacian",
            "fine_n": a.fine_n,
            "coarse_n": a.coarse_n,
            "lx": a.lx,
            "ly": a.ly,
            "base_channels": a.base_channels,
            "lambda_dns": a.lambda_dns,
            "lambda_spectrum": a.lambda_spectrum,
        }
        torch.save(c, ckdir / "latest.pt")
        if vm["total"] < best:
            best = vm["total"]
            torch.save(c, ckdir / "best.pt")
        if a.save_every > 0 and epoch % a.save_every == 0:
            torch.save(c, ckdir / f"epoch_{epoch:05d}.pt")
        with (a.output_dir / "training_history.csv").open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(history[0]))
            w.writeheader()
            w.writerows(history)
        print(
            f"Epoch {epoch:05d} | train={tm['total']:.6e} | "
            f"validation={vm['total']:.6e} | "
            f"DNS relL2={vm['dns_relative_l2']:.6e}"
        )
    c = load_checkpoint(ckdir / "best.pt", device)
    model.load_state_dict(c["model"])
    test = run_epoch(
        model,
        test_loader,
        device,
        mean,
        std,
        alpha,
        a.lx,
        a.ly,
        a.lambda_dns,
        a.lambda_spectrum,
    )
    for name, value in test.items():
        writer.add_scalar(f"metrics/test/{name}", value, int(c["epoch"]))
    writer.flush()
    writer.close()
    (a.output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "dataset": str(a.dataset),
                "train_snapshots": len(tr),
                "validation_snapshots": len(va),
                "test_snapshots": len(te),
                "best_epoch": int(c["epoch"]),
                "test_metrics": test,
                "tensorboard_dir": str(tensorboard_dir),
            },
            indent=2,
        )
    )

    print("Test metrics:", test)


if __name__ == "__main__":
    main()
