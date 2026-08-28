# Germano U-Net for Hasegawa--Wakatani

This directory contains the training, standalone inference, diagnostics, and
BOUT++ coupling code for reconstructing fine-grid Hasegawa--Wakatani fields
from Germano-filtered LES fields.

## Environment

Activate an environment containing PyTorch, NumPy, Matplotlib, and TensorBoard.
For the current system this is normally:

```bash
source /gpfs/mcc/HT04543/jxc06/jxc74-jxc06/programs/miniconda3/etc/profile.d/conda.sh
conda activate py311
```

## Training

`train_germano_unet.py` recursively reads `.npz` DNS snapshots from the
dataset directory. A dataset may either contain all snapshots in one tree, in
which case the script creates the train/validation/test split, or contain
explicit `train/`, `validation/`, and `test/` subdirectories.

The supplied `run.sh` is a SLURM example. Its principal settings can be
overridden when submitting:

```bash
DNS_N=256 LES_N=128 OUTPUT_DIR=germano_training_output_dns256_les128 sbatch run.sh
```

An equivalent direct command is:

```bash
python train_germano_unet.py \
  --dataset ../../data/BOUT_runs/HW_2D/Papers/PoP23/HW_N256/fields/ \
  --output-dir germano_training_output_dns256_les128 \
  --fine-n 256 --coarse-n 128 \
  --lx 50.176 --ly 50.176 \
  --alpha 0.392 \
  --epochs 100 --batch-size 4 \
  --precision float64 --no-normalize \
  --inverse-cutoff-fraction 0.707107 \
  --inverse-cutoff-order 8 \
  --post-filter none
```

`--alpha` is the physical Germano filter width. If omitted, it defaults to
`Lx/fine_n`. The fine resolution must be divisible by the coarse resolution.
Training with the default `--vorticity-from-potential` makes the U-Net predict
`U` and `V`; `P` (vorticity) is then obtained from the periodic Laplacian of
`V`. Use `--resume PATH/TO/checkpoint.pt` to resume a compatible run.

Important output files are:

- `checkpoints/best.pt`: checkpoint with the best validation loss.
- `checkpoints/latest.pt`: most recent checkpoint.
- `checkpoints/epoch_XXXXX.pt`: periodic checkpoints.
- `training_parameters.json` and `metadata.json`: run configuration and
  metadata.
- `training_history.csv`: losses and metrics by epoch.
- `training.log`: captured console output.
- `tensorboard/`: TensorBoard event files.

## TensorBoard

Point TensorBoard at either one run or a directory containing several runs:

```bash
tensorboard --logdir germano_training_output_dns256_les128/tensorboard
```

Then open the URL printed by TensorBoard, normally
`http://localhost:6006`. On a remote cluster, use SSH port forwarding or the
site's supported remote-visualization method. A custom event directory can be
selected during training with `--tensorboard-dir`.

## Standalone inference

Run inference on one DNS `.npz` snapshot with:

```bash
python infer_germano_unet.py \
  --checkpoint germano_training_output_dns256_les128/checkpoints/best.pt \
  --input fields_HW_run0_time501.npz \
  --output-dir germano_inference_output
```

The default `--inverse-input unet` passes the U-Net output to the inverse
Germano operation. `--inverse-input filtered-dns` bypasses the U-Net output
for that operation and is useful as an inverse-filter baseline. The inference
output directory is recreated on every run, so do not point it at a directory
containing files that must be retained.

Inference accepts the same runtime spectral regularization overrides used by
the BOUT++ interface:

```bash
export PBOUT_UNET_INVERSE_CUTOFF_FRACTION=0.707107
export PBOUT_UNET_INVERSE_CUTOFF_ORDER=8
export PBOUT_UNET_POST_FILTER=none        # none, high_order, or gaussian
export PBOUT_UNET_POST_CUTOFF_FRACTION=none
export PBOUT_UNET_POST_CUTOFF_ORDER=8
```

An unset override uses the checkpoint value. `none` disables an optional
cutoff. The inverse cutoff is applied before inverse Germano reconstruction;
the post filter/cutoff is applied afterward.

## BOUT++ coupling

`pBOUT-unet.py` exposes `initFlow()` and `findLESTerms()` to the embedded
Python interface used by the Hasegawa--Wakatani BOUT++ executable. A typical
launch configuration is:

```bash
export PBOUT_UNET_DIR="$PWD"
export PBOUT_UNET_CHECKPOINT="$PWD/germano_training_output_dns256_les128/checkpoints/best.pt"
export PBOUT_UNET_DNS_PATH="$PWD/fields_HW_run0_time501.npz"
export PBOUT_UNET_DNS_INIT=1
export PBOUT_UNET_RECONSTRUCT_CHANNELS=U
export PBOUT_UNET_ALPHA=0.392
export PBOUT_UNET_DEVICE=cuda
export PBOUT_UNET_WRITE_DIAGNOSTICS=0     # disable images and saved fields
export PBOUT_UNET_INVERSE_CUTOFF_FRACTION=0.707107
export PBOUT_UNET_INVERSE_CUTOFF_ORDER=8

/path/to/hasegawa-wakatani
```

`PBOUT_UNET_ALPHA` overrides the checkpoint's physical Germano width. The
Arakawa terms are Germano-filtered when `ALPHA > 0`, except in
`PBOUT_UNET_DNS_INIT=2`, which always uses unfiltered DNS fields and terms.
Set `PBOUT_UNET_WRITE_DIAGNOSTICS=0` to suppress all pBOUT++ diagnostic image
and field files, including initial diagnostics, snapshots, and animations.

### Initialization modes and required BOUT++ grid

| `PBOUT_UNET_DNS_INIT` | Initialization and runtime behavior | BOUT++ active grid |
| --- | --- | --- |
| `0` | Germano-filter the DNS snapshot, downsample it, evolve the coarse LES fields, reconstruct fine fields with the U-Net/inverse Germano operation, calculate fine-grid Arakawa terms, filter them when `ALPHA > 0`, and downsample the terms. | `coarse_n` |
| `1` with reconstruction channels | Germano-filter and downsample the initial DNS. Reconstruct only the selected `U,V` channels; unselected channels use the spectrally resized incoming fields. Apply inverse Germano reconstruction, calculate fine-grid Arakawa terms, filter them when `ALPHA > 0`, and downsample the terms. | `coarse_n` |
| `1` with `none` | Keep the Germano-filtered initial fields on the fine grid. No U-Net is used, but inverse Germano reconstruction and its configured cutoffs are still applied before the Arakawa calculation. The resulting terms are Germano-filtered when `ALPHA > 0`. | `fine_n` |
| `2` | Use and evolve the original DNS snapshot directly. No U-Net, inverse Germano reconstruction, or Germano filtering of the Arakawa terms is used. | `fine_n` |

BOUT++ `nx` includes four X guard cells, while `nz` has no periodic guard
cells. For the 256-to-128 checkpoint, use the following for a coarse-grid run
(mode 0, or mode 1 with selected reconstruction channels):

```ini
[mesh]
Lx = 50.176
Lz = 50.176
nx = 132  # 128 active points + 4 guard cells
nz = 128
```

For a fine-grid run (mode 1 with `none`, or mode 2), use:

```ini
[mesh]
Lx = 50.176
Lz = 50.176
nx = 260  # 256 active points + 4 guard cells
nz = 256
```

The domain lengths in `BOUT.inp` must match the checkpoint. A mismatch in
active resolution or grid spacing is rejected by the Python interface.

### Reconstruction channels

`PBOUT_UNET_RECONSTRUCT_CHANNELS` defaults to `none` and accepts `none`,
`all`, or a comma-separated subset such as `U` or `U,V`. With the current
two-output checkpoints, `P` is not independently predicted: it is computed as
the configured periodic Laplacian of reconstructed `V`. Consequently `U,V`
and `U,V,P` have the same effective reconstruction behavior.

Channel selection is effective in mode 1. In the current mode-0 path, the
U-Net's complete output is used, so selecting only `U` still reconstructs both
model outputs (`U` and `V`). Mode 2 ignores reconstruction-channel selection.

Other useful controls are:

```bash
export PBOUT_UNET_IMPLICIT=1
export PBOUT_UNET_OMEGA_FROM_POTENTIAL=1
export PBOUT_UNET_LAPLACIAN_ORDER=2       # 2 or 4
export PBOUT_UNET_POST_FILTER=none        # none, high_order, gaussian
export PBOUT_UNET_POST_CUTOFF_FRACTION=none
export PBOUT_UNET_POST_CUTOFF_ORDER=8
```

At startup, `pBOUT-unet.py` prints checkpoint metadata and the effective
runtime configuration. Check this output first when diagnosing checkpoint,
resolution, alpha, precision, or filtering mismatches.
