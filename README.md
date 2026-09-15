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

## U-Net with bicubic input interpolation

Training uses `LegacyResidualUNetSR`. Periodic bicubic interpolation upsamples LES
fields to the DNS grid before the U-Net encoder. Encoder–decoder skip
connections remain, and the decoder uses periodic bilinear interpolation.
The default model does not add the interpolated input directly to its output.

The pipeline is **LES → NN at DNS resolution → configured inverse-input
filter → Germano inversion**. Training compares reconstructed DNS density and potential with DNS targets
using mean absolute error scaled by DNS training standard deviations, plus
an optional normalized DNS vorticity error. Gradients pass through inversion.
The loss adds squared normalized DNS errors to the L1 terms with
`--lambda-mse 0.1` by default, including vorticity scaled by
`--lambda-vorticity`. Use `--lambda-mse 0` for the L1 objective from
commit `c52b744`. KL regularization is disabled by default.

Variational heads are disabled by default; use `--variational` to enable
them. `--lambda-kl` enables and controls the optional KL penalty (default: 0). Evaluation uses the latent mean.

Checkpoints record `architecture: LegacyResidualUNetSR`. Earlier `BicubicUNetSR`, `ResidualUNetSR`,
`PeriodicResidualCNN`, and `PeriodicSubpixelCNN` checkpoints remain supported.
The small bicubic CNN implementation has been removed; its saved training
results remain available, but its checkpoints cannot be loaded by this code.

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
  --inverse-filter filtered_dns_spectrum \
  --inverse-cutoff-fraction 0.707107 \
  --inverse-cutoff-order 8 \
  --post-filter none
```

`--alpha` is the physical Germano filter width. If omitted, it defaults to
`Lx/fine_n`. The fine resolution must be divisible by the coarse resolution.
Training with the default `--vorticity-from-potential` makes the U-Net predict
`U` and `V`; `P` (vorticity) is then obtained from the periodic Laplacian of
`V`. Use `--resume PATH/TO/checkpoint.pt` to resume a compatible run.

`--inverse-filter filtered_dns_spectrum` computes the mean radial Fourier
power of the actual filtered DNS over the training split before epoch 1. The
reference is stored in the checkpoint, and the pre-inversion filter attenuates
only U-Net shells whose power exceeds it. Set `--inverse-min-wavenumber 2.0`
to leave physical wavenumbers `k <= 2.0` unchanged. The default is `1.0`;
`0` applies spectrum matching to all nonzero modes. This threshold is in
physical angular wavenumber units (`k_x = 2*pi*m/Lx`), not a Nyquist fraction.
It affects only `filtered_dns_spectrum`; `--inverse-cutoff-fraction` retains
its separate meaning for `cutoff`.

With the SLURM wrapper, use `INVERSE_MIN_WAVENUMBER=2.0 sbatch run.sh`.
The threshold is saved in checkpoints. Standalone inference accepts
`--inverse-min-wavenumber 2.0` to override it; BOUT++ accepts
`PBOUT_UNET_INVERSE_MIN_WAVENUMBER=2.0`. Old checkpoints default to `1.0`.
 The default `cutoff` retains the
existing smooth exponential cutoff and its fraction/order options.

Important output files are:

- `checkpoints/best.pt`: checkpoint with the best validation loss.
- `checkpoints/latest.pt`: most recent checkpoint.
- `checkpoints/epoch_XXXXX.pt`: periodic checkpoints.
- `training_parameters.json`: readable export of the configuration embedded in
  checkpoints from the current training invocation.
- `metadata.json`: run status, best epoch, test results, and report locations.
  It contains no model or training configuration in newly completed runs.
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
export PBOUT_UNET_INVERSE_FILTER=filtered_dns_spectrum # or cutoff
export PBOUT_UNET_POST_FILTER=none        # none, high_order, or gaussian
export PBOUT_UNET_POST_CUTOFF_FRACTION=none
export PBOUT_UNET_POST_CUTOFF_ORDER=8
```

An unset override uses the checkpoint value. `none` disables an optional
cutoff. The selected inverse filter is applied before inverse Germano
reconstruction; the post filter/cutoff is applied afterward. The
`filtered_dns_spectrum` setting requires a checkpoint trained with that option.

## BOUT++ coupling

Set `PBOUT_UNET_RESCALING` to select density/potential rescaling:

- `0` (default): no rescaling.
- `1`: formula 11, the median pointwise correction described below.
- `2`: formula 13, `DNS_max_new = LES_max_new * DNS_max_previous / LES_max_previous`.
  This uses the actual reconstructed DNS and LES absolute maxima from the
  preceding solver time. The initial DNS/LES seed the first call. Repeated
  calls at the same time reuse the preceding time's ratio; time rollback
  discards later stored values.
  A zero reference LES maximum or zero reconstruction retains the amplitude;
  a zero current LES requests zero amplitude.

- `3`: keep each reconstructed density/potential absolute maximum equal to
  its initial DNS absolute maximum at simulation time `t=0`. An identically
  zero reconstruction stays zero because multiplicative scaling cannot restore it.

With option `1`, DNS reconstruction uses equation (11) with
one amplitude estimate per batch and field:
`DNS_max_new = median((LES / LES_max) / G(DNS / DNS_max)) * LES_max`,
then `DNS_corrected = (DNS / DNS_max) * DNS_max_new`.
Here `LES_max` and `DNS_max` are the respective absolute spatial maxima.
The normalized DNS is Germano-filtered and spectrally resized to the incoming
LES grid before taking the ratio. For this linear filter, normalization
cancels algebraically, giving the same correction as the unnormalized ratio.
Nonfinite ratios and near-zero denominators are excluded. If no valid ratio
remains, the amplitude is retained; an identically zero LES requests zero
amplitude. Checkpoint mean/std normalization is unchanged.
Vorticity is then recomputed from the corrected potential when configured.
Rescaling applies to U-Net and inverse-only reconstruction; unfiltered DNS and
DNS replay retain their original amplitudes.

The interface selects packages matching the embedded Python: Python 3.13 uses
`~/.pyenvs/BOUT-Hermes3/lib/python3.13/site-packages`, and Python 3.11 uses the
local conda `py311` installation. Set `PBOUT_UNET_SITE_PACKAGES` to override
this directory with an installation for the same Python major/minor version.
An empty override uses the interpreter's existing package paths. Activating
a virtual environment does not change BOUT++'s linked Python library.

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
export PBOUT_UNET_WRITE_DIAGNOSTICS=2     # save only results_U-Net/fields
export PBOUT_UNET_INVERSE_CUTOFF_FRACTION=0.707107
export PBOUT_UNET_INVERSE_CUTOFF_ORDER=8

/path/to/hasegawa-wakatani
```

`PBOUT_UNET_ALPHA` overrides the checkpoint's physical Germano width. The
Arakawa terms are Germano-filtered when `ALPHA > 0`, except in
`PBOUT_UNET_DNS_INIT=2`, which always uses unfiltered DNS fields and terms.
`PBOUT_UNET_WRITE_DIAGNOSTICS` accepts three modes:

| Value | Diagnostic output |
| --- | --- |
| `0` | Disable all diagnostic files. |
| `1` | Write all field, image, Arakawa, and animation diagnostics (default). |
| `2` | Write only field snapshots under `results_U-Net/fields`. |

At the start of every fresh BOUT++ inference, pBOUT removes any existing
`results_U-Net` directory, even when diagnostic writing is disabled.

### Initialization modes and required BOUT++ grid

`PBOUT_UNET_DNS_INIT` accepts `0`, `1`, `2`, or `3`.
Any other value causes pBOUT to stop with a configuration error.

| `PBOUT_UNET_DNS_INIT` | Initialization and runtime behavior | BOUT++ active grid |
| --- | --- | --- |
| `0` | Germano-filter the DNS snapshot, downsample it, evolve the coarse LES fields, reconstruct fine fields with the U-Net/inverse Germano operation, calculate fine-grid Arakawa terms, filter them when `ALPHA > 0`, and downsample the terms. | `coarse_n` |
| `1` with reconstruction channels | Germano-filter and downsample the initial DNS. Reconstruct only the selected `U,V` channels; unselected channels use the spectrally resized incoming fields. Apply inverse Germano reconstruction, calculate fine-grid Arakawa terms, filter them when `ALPHA > 0`, and downsample the terms. | `coarse_n` |
| `1` with `none` | Keep the Germano-filtered initial fields on the fine grid. No U-Net is used, but inverse Germano reconstruction and its configured cutoffs are still applied before the Arakawa calculation. The resulting terms are Germano-filtered when `ALPHA > 0`. | `fine_n` |
| `2` | Use and evolve the original DNS snapshot directly. No U-Net, inverse Germano reconstruction, or Germano filtering of the Arakawa terms is used. | `fine_n` |

Mode `3` uses `coarse_n` active points, like mode `0`, and ignores reconstruction-channel selection.

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
export PBOUT_UNET_RESCALING=0            # 0=off, 1=formula 11, 2=formula 13, 3=initial DNS maxima
export PBOUT_UNET_OMEGA_FROM_POTENTIAL=1  # optional override of checkpoint default
export PBOUT_UNET_LAPLACIAN_ORDER=2       # 2 or 4
export PBOUT_UNET_POST_FILTER=none        # none, high_order, gaussian
export PBOUT_UNET_POST_CUTOFF_FRACTION=none
export PBOUT_UNET_POST_CUTOFF_ORDER=8
```

At startup, `pBOUT-unet.py` prints checkpoint metadata and the effective
runtime configuration. Check this output first when diagnosing checkpoint,
resolution, alpha, precision, or filtering mismatches.

Set `PBOUT_UNET_DNS_INIT=3` to compute Arakawa terms from saved DNS fields
instead of reconstructing them. `PBOUT_UNET_DNS_FIELD_DIR` defaults to
`results_U-Net_DNS/fields`; override it if your snapshots are stored elsewhere
(for example, in a `fields` directory). Each `.npz` must contain `U`, `V`, `P`
and a scalar physical timestamp `simtime`. The fields are linearly interpolated
between the two snapshots surrounding the current BOUT++ physical time, then
used to calculate the Arakawa brackets. Saved `P` is used directly, without
replacing it with a Laplacian of `V`. Times outside the saved range are rejected.

Mode 3 initializes exactly as mode 0: load `PBOUT_UNET_DNS_PATH`, apply the
Germano filter, and downsample to the coarse LES grid. During each callback,
it reads and interpolates the saved DNS fields instead of reconstructing them,
calculates fine-grid Arakawa brackets, applies the Germano filter when `ALPHA > 0`,
and downsamples the brackets to the coarse grid. The series must cover time zero
for initial diagnostics and every requested simulation time. The checkpoint
still supplies grid/domain and filter configuration.

`PBOUT_UNET_IMPLICIT` independently accepts only `0` or `1` (default `1`):
`0` returns the filtered, downsampled DNS brackets in mode 3; `1` returns
the incoming BOUT++ brackets minus those filtered, downsampled brackets. This must match `implicitStylES` in `hw.cxx` (`false` for `0`,
`true` for `1`), which determines whether BOUT++ passes three or five channels.

In replay mode (`PBOUT_UNET_DNS_INIT=3`), set `PBOUT_UNET_WRITE_DIAGNOSTICS=1`
to also reconstruct DNS with the U-Net at each diagnostic output time and save
comparisons under `results_U-Net/images`. Files named
`fields_unet_vs_interpolated_time*.png` compare U/V/P, while
`arakawa_dns_unet_vs_interpolated_time*.png` compare fine-grid brackets and
`arakawa_les_unet_vs_interpolated_time*.png` compare filtered, downsampled brackets
before implicit subtraction. Each image shows the two sources and their difference.
This adds diagnostic reconstruction work; the solver still uses interpolated DNS.

Replay comparisons also save `arakawa_les_spectra_time*.png`, showing the
U-Net and interpolated-DNS spectra of both filtered/downsampled Arakawa terms
before implicit subtraction, with fixed axes across frames. The x-axis covers
all positive wavenumber shells of the DNS grid (`fine_n`); the plotted curves
still represent filtered/downsampled LES terms. At the final BOUT++
output these are assembled into `results_U-Net/images/arakawa_les_spectra_animation.gif`
(with separate batch suffixes when applicable). This requires diagnostic mode `1`.

Inference configuration comes from the selected `.pt` checkpoint, including its
embedded training parameters. Neither JSON report supplies inference settings;
resuming training may replace the readable export while older checkpoints retain
their own configuration. Existing checkpoints and older metadata files remain
supported. Legacy configuration entries in metadata are ignored at startup.

By default, pBOUT preserves the checkpoint's DNS vorticity convention. Explicit
`PBOUT_UNET_*` overrides of checkpoint settings are printed with their checkpoint,
requested, and effective values and whether they changed the setting. A model
that outputs only density and potential still requires derived vorticity even if
an environment override requests otherwise. Runtime controls such as input paths
and diagnostic modes are shown in `effective_inference`.
