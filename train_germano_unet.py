#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, json, random
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
import sys
import traceback
import numpy as np, torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from germano_common import *


class Tee:
    """Write a text stream to both the terminal and a log file."""

    def __init__(self, terminal, log_file):
        self.terminal = terminal
        self.log_file = log_file

    def write(self, text):
        self.terminal.write(text)
        self.log_file.write(text)
        return len(text)

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def isatty(self):
        return self.terminal.isatty()


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
    dns_std,
    alpha,
    lx,
    ly,
    lambda_dns,
    lambda_spectrum,
    inverse_cutoff_fraction,
    inverse_cutoff_order,
    learn_vorticity,
    laplacian_order,
    optimizer=None,
):
    training = optimizer is not None
    model.train(training)
    loss_names = [
        "total",
        "filtered",
        "dns",
        "spectrum",
        "dns_U",
        "dns_V",
        "dns_W",
    ]
    totals = {
        name: 0.0
        for name in loss_names
        + [
            "filtered_relative_l2",
            "dns_relative_l2",
            "spectrum_relative_l2",
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
                dns_std,
                alpha,
                lx,
                ly,
                lambda_dns,
                lambda_spectrum,
                inverse_cutoff_fraction,
                inverse_cutoff_order,
                learn_vorticity,
                laplacian_order,
            )
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            b = coarse.shape[0]
            count += b
            for k in loss_names:
                totals[k] += vals[k].detach().item() * b
            totals["spectrum_relative_l2"] += (
                vals["spectrum_relative_l2"].detach().item() * b
            )

            totals["filtered_relative_l2"] += (
                relative_l2(ft * std + mean, pf).detach().item() * b
            )
            totals["dns_relative_l2"] += (
                relative_l2(dns, pd).detach().item() * b
            )

    return {k: v / count for k, v in totals.items()}


def configure_tensorboard(writer, normalize):
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
                for name in (
                    "filtered_relative_l2",
                    "dns_relative_l2",
                    "spectrum_relative_l2",
                )
            },
            (
                "Per-field normalized DNS loss"
                if normalize
                else "Per-field absolute DNS loss"
            ): {
                name: [
                    "Multiline",
                    [
                        f"metrics/train/dns_{name}",
                        f"metrics/validation/dns_{name}",
                    ],
                ]
                for name in HW_FIELD_NAMES
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


def serialise_parameters(args, **derived):
    parameters = {}
    for name, value in vars(args).items():
        if isinstance(value, Path):
            value = str(value)
        parameters[name] = value
    parameters.update(derived)
    return parameters


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
    p.add_argument(
        "--precision",
        choices=("float32", "float64"),
        default="float32",
        help="Floating-point precision for data, U-Net, losses, and physics",
    )
    p.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Normalize LES inputs and filtered targets and scale DNS loss "
            "by its standard deviation (default: enabled; use "
            "--no-normalize for physical values throughout)"
        ),
    )
    p.add_argument(
        "--learn-vorticity",
        action="store_true",
        help=(
            "Predict vorticity as an independent third U-Net output channel "
            "instead of deriving it as the Laplacian of potential"
        ),
    )
    p.add_argument(
        "--laplacian-order",
        type=int,
        choices=(2, 4),
        default=2,
        help=(
            "Centered periodic Laplacian order used to derive vorticity "
            "from potential (default: 2)"
        ),
    )
    p.add_argument("--lambda-dns", type=float, default=0.05)
    p.add_argument("--lambda-spectrum", type=float, default=0.02)
    p.add_argument(
        "--inverse-cutoff-fraction",
        type=float,
        default=0.75,
        help="Smooth inverse cutoff as a fraction of Nyquist; 0 disables it",
    )
    p.add_argument("--inverse-cutoff-order", type=float, default=8.0)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--resume", type=Path)
    p.add_argument(
        "--tensorboard-dir",
        type=Path,
        help="TensorBoard log directory (default: OUTPUT_DIR/tensorboard)",
    )
    p.add_argument(
        "--log-file",
        type=Path,
        help="Console log file (default: OUTPUT_DIR/training.log)",
    )
    return p.parse_args()


def run_training(a):
    set_seed(a.seed)
    torch_dtype = getattr(torch, a.precision)
    numpy_dtype = getattr(np, a.precision)
    if tuple(a.fields) != HW_FIELD_NAMES:
        raise ValueError(f"Hasegawa-Wakatani fields must be {HW_FIELD_NAMES}.")
    if a.fine_n != 8 * a.coarse_n:
        raise ValueError("Current model requires fine_n = 8 * coarse_n.")
    if not 0 <= a.inverse_cutoff_fraction <= 1:
        raise ValueError("inverse-cutoff-fraction must be in [0, 1].")
    if a.inverse_cutoff_order <= 0:
        raise ValueError("inverse-cutoff-order must be positive.")
    inverse_cutoff_fraction = a.inverse_cutoff_fraction or None
    a.output_dir.mkdir(parents=True, exist_ok=True)
    ckdir = a.output_dir / "checkpoints"
    ckdir.mkdir(exist_ok=True)
    parameters_path = a.output_dir / "training_parameters.json"
    parameters_path.write_text(
        json.dumps(serialise_parameters(a), indent=2, sort_keys=True)
    )
    tr, va, te = split_files(
        a.dataset, a.validation_fraction, a.test_fraction, a.split_mode, a.seed
    )
    delta = a.lx / a.fine_n
    alpha = a.alpha if a.alpha is not None else a.alpha_factor * delta
    if a.normalize:
        mean_np, std_np, dns_std_np = calculate_training_statistics(
            tr,
            a.fields,
            a.fine_n,
            a.coarse_n,
            alpha,
            a.lx,
            a.ly,
            dtype=numpy_dtype,
            laplacian_order=a.laplacian_order,
        )
    else:
        statistics_shape = (1, len(a.fields), 1, 1)
        mean_np = np.zeros(statistics_shape, dtype=numpy_dtype)
        std_np = np.ones(statistics_shape, dtype=numpy_dtype)
        dns_std_np = np.ones(statistics_shape, dtype=numpy_dtype)
        print(
            "Normalization disabled: LES inputs, filtered targets, model "
            "outputs, reconstructed DNS, and DNS losses use physical values."
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
            a.laplacian_order,
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
    print("Precision:", a.precision)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))
        torch.backends.cudnn.benchmark = True
    model_output_channels = 3 if a.learn_vorticity else 2
    model = ResidualUNetSR(
        len(a.fields),
        a.base_channels,
        a.fine_n // a.coarse_n,
        model_output_channels,
    ).to(device=device, dtype=torch_dtype)
    opt = torch.optim.AdamW(
        model.parameters(), lr=a.learning_rate, weight_decay=1e-6
    )
    mean = torch.from_numpy(mean_np).to(device)
    std = torch.from_numpy(std_np).to(device)
    dns_std = torch.from_numpy(dns_std_np).to(device)
    start = 1
    best = float("inf")
    if a.resume:
        c = load_checkpoint(a.resume, device)
        validate_hw_checkpoint(c)
        checkpoint_learns_vorticity = (
            c.get("vorticity_mode", "potential_laplacian")
            == "learned_channel"
        )
        if checkpoint_learns_vorticity != a.learn_vorticity:
            raise ValueError(
                "--learn-vorticity must match the resumed checkpoint."
            )
        checkpoint_operator = c.get(
            "vorticity_operator",
            c.get("training_parameters", {}).get(
                "vorticity_operator", "fourth_order_periodic_laplacian"
            ),
        )
        checkpoint_order = (
            int(c["laplacian_order"])
            if "laplacian_order" in c
            else 2
            if checkpoint_operator == "second_order_periodic_laplacian"
            else 4
        )
        if not a.learn_vorticity and checkpoint_order != a.laplacian_order:
            raise ValueError(
                f"--laplacian-order {a.laplacian_order} does not match "
                f"resumed checkpoint order {checkpoint_order}."
            )
        checkpoint_precision = c.get("precision", "float32")
        if checkpoint_precision != a.precision:
            raise ValueError(
                f"--precision {a.precision} does not match resumed "
                f"checkpoint precision {checkpoint_precision}."
            )
        checkpoint_normalizes = c.get(
            "normalize",
            c.get("training_parameters", {}).get("normalize", True),
        )
        if checkpoint_normalizes != a.normalize:
            raise ValueError(
                f"--{'normalize' if a.normalize else 'no-normalize'} does "
                "not match the resumed checkpoint."
            )
        model.load_state_dict(c["model"])
        opt.load_state_dict(c["optimizer"])
        start = int(c["epoch"]) + 1
        best = float(c["validation_loss"])
        print("Resumed from epoch", start - 1)
    tensorboard_dir = a.tensorboard_dir or a.output_dir / "tensorboard"
    writer = SummaryWriter(log_dir=tensorboard_dir, purge_step=start)
    configure_tensorboard(writer, a.normalize)
    print("TensorBoard logs:", tensorboard_dir)
    parameters = serialise_parameters(
        a,
        effective_alpha=alpha,
        grid_spacing=delta,
        scale_factor=a.fine_n // a.coarse_n,
        optimizer="AdamW",
        optimizer_learning_rate=opt.param_groups[0]["lr"],
        optimizer_weight_decay=opt.param_groups[0]["weight_decay"],
        optimizer_betas=opt.param_groups[0]["betas"],
        optimizer_epsilon=opt.param_groups[0]["eps"],
        optimizer_amsgrad=opt.param_groups[0]["amsgrad"],
        gradient_clip_norm=1.0,
        model="ResidualUNetSR",
        model_input_channels=len(a.fields),
        model_output_channels=model_output_channels,
        trainable_model_parameters=sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ),
        differential_filter="1 / (1 + alpha^2 k^2)",
        downsampling="centered Fourier-mode truncation",
        vorticity_operator=(
            "learned_independent_channel"
            if a.learn_vorticity
            else "second_order_periodic_laplacian"
            if a.laplacian_order == 2
            else "fourth_order_periodic_laplacian"
        ),
        vorticity_mode=(
            "learned_channel" if a.learn_vorticity else "potential_laplacian"
        ),
        convolution_padding="circular",
        interpolation_boundary="periodic",
        inverse_regularization="smooth_exponential_cutoff",
        inverse_cutoff_wavenumber=(
            inverse_cutoff_fraction * np.pi / delta
            if inverse_cutoff_fraction is not None
            else None
        ),
        physics_version=1,
        train_snapshots=len(tr),
        validation_snapshots=len(va),
        test_snapshots=len(te),
        normalization_mean=mean_np.reshape(-1).tolist(),
        normalization_std=std_np.reshape(-1).tolist(),
        dns_normalization_std=dns_std_np.reshape(-1).tolist(),
        normalization_enabled=a.normalize,
        device=str(device),
        gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        tensorboard_dir=str(tensorboard_dir),
    )
    parameters_path.write_text(
        json.dumps(parameters, indent=2, sort_keys=True)
    )
    history = []
    for epoch in range(start, a.epochs + 1):
        tm = run_epoch(
            model,
            train_loader,
            device,
            mean,
            std,
            dns_std,
            alpha,
            a.lx,
            a.ly,
            a.lambda_dns,
            a.lambda_spectrum,
            inverse_cutoff_fraction,
            a.inverse_cutoff_order,
            a.learn_vorticity,
            a.laplacian_order,
            opt,
        )
        vm = run_epoch(
            model,
            val_loader,
            device,
            mean,
            std,
            dns_std,
            alpha,
            a.lx,
            a.ly,
            a.lambda_dns,
            a.lambda_spectrum,
            inverse_cutoff_fraction,
            a.inverse_cutoff_order,
            a.learn_vorticity,
            a.laplacian_order,
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
            "dns_std": dns_std_np,
            "field_names": list(a.fields),
            "physics_version": 1,
            "vorticity_operator": (
                "learned_independent_channel"
                if a.learn_vorticity
                else "second_order_periodic_laplacian"
                if a.laplacian_order == 2
                else "fourth_order_periodic_laplacian"
            ),
            "laplacian_order": a.laplacian_order,
            "vorticity_mode": (
                "learned_channel"
                if a.learn_vorticity
                else "potential_laplacian"
            ),
            "model_output_channels": model_output_channels,
            "training_parameters": parameters,
            "fine_n": a.fine_n,
            "coarse_n": a.coarse_n,
            "lx": a.lx,
            "ly": a.ly,
            "base_channels": a.base_channels,
            "lambda_dns": a.lambda_dns,
            "lambda_spectrum": a.lambda_spectrum,
            "inverse_cutoff_fraction": inverse_cutoff_fraction,
            "inverse_cutoff_order": a.inverse_cutoff_order,
            "precision": a.precision,
            "normalize": a.normalize,
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
        dns_std,
        alpha,
        a.lx,
        a.ly,
        a.lambda_dns,
        a.lambda_spectrum,
        inverse_cutoff_fraction,
        a.inverse_cutoff_order,
        a.learn_vorticity,
        a.laplacian_order,
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
                "training_parameters_file": str(parameters_path),
                "training_parameters": parameters,
            },
            indent=2,
        )
    )

    print("Test metrics:", test)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.log_file = args.log_file or args.output_dir / "training.log"
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    with args.log_file.open("a", buffering=1) as log_file:
        stdout = Tee(sys.stdout, log_file)
        stderr = Tee(sys.stderr, log_file)
        with redirect_stdout(stdout), redirect_stderr(stderr):
            print(f"Training console log: {args.log_file}")
            try:
                run_training(args)
            except Exception:
                traceback.print_exc()
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
