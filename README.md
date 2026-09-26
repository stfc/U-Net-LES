# Germano U-Net for Hasegawa--Wakatani

This directory contains the training, standalone inference, diagnostics, and
BOUT++ coupling code for reconstructing fine-grid Hasegawa--Wakatani fields
from Germano-filtered LES fields.

## Reproduce the results

Two routes lead to the same BOUT++ time-integration workflow:

1. **Faster: use pretrained weights.** Download a matching checkpoint from
   [Zenodo (DOI: 10.5281/zenodo.22973917)](https://doi.org/10.5281/zenodo.22973917),
   then run the coupled solver as described in [Option 1](#option-1-pretrained-weights-and-bout-time-integration).
2. **Train your own model.** Train on DNS snapshots, inspect the learning curves
   with `plot_tensorboard.py`, check one snapshot with `infer_germano_unet.py`,
   and use the resulting checkpoint in the same solver workflow. See
   [Option 2](#option-2-train-check-and-integrate).

The worked example below is **HW, DNS 512² → LES 128²**, using the model and
physical input in `full_runs/HW_N512`. Run commands from the repository root
unless a subshell explicitly changes directory.

### Environment and inputs

Use Python 3.10 or newer with PyTorch, NumPy, Matplotlib, TensorBoard, Pillow
and netCDF4. For example, in a Python environment of your choice:

```bash
python -m pip install torch numpy matplotlib tensorboard pillow netCDF4
export UNET_ROOT="$PWD"
```

For CUDA runs, use a PyTorch build compatible with your machine's CUDA runtime.
A CPU can be used for inference by setting `PBOUT_UNET_DEVICE=cpu`.

**Time integration requires an installed or already built BOUT++ library**,
its CMake package (`bout++Config.cmake`), a compatible C/C++ compiler and MPI
installation, and Python development headers/library. Training and standalone
snapshot inference do not require BOUT++. The executable built here embeds
Python and imports **`pBOUT-unet.py`** (hyphen, not `pBOUT_unet.py`); do not
launch that callback module as a standalone simulation script.

You also need a DNS initial condition matching the checkpoint's physical case,
fine-grid resolution and domain. The worked example uses
`fields/fields_HW_run0_time381_N512.npz`. Snapshots store `U` (density), `V`
(potential) and `P` (vorticity). The pretrained weights alone are not initial
conditions or a training dataset. If the snapshot is absent from your checkout,
obtain the matching DNS snapshot before running this example.

### Option 1: pretrained weights and BOUT++ time integration

#### Download and select weights

Download and extract the weights from the
[Zenodo record](https://doi.org/10.5281/zenodo.22973917). Select the **HW,
`fine_n=512`, `coarse_n=128`** checkpoint for this example. Set its actual local
path below; the command does not assume a particular archive layout.

```bash
export PBOUT_UNET_CHECKPOINT="/absolute/path/to/downloaded/HW_dns512_les128/best.pt"
export PBOUT_UNET_DNS_PATH="$UNET_ROOT/fields/fields_HW_run0_time381_N512.npz"
```

The checkpoint supplies architecture, domain lengths, Germano filter width,
normalization and inverse-filter settings. Keep those defaults for the first
run; do not apply overrides copied from a different training experiment. The
startup log reports the effective configuration. In this example the domain
is `Lx=Lz=50.176`; the Germano filter width is distinct from the physical
adiabaticity parameter `[hw] alpha=1.0` in `BOUT.inp`.

#### Compile the coupled model

Activate the compiler/MPI environment used to build BOUT++ first. The default
BOUT++ location is **`../BOUT_repo` relative to this repository**, with CMake
searching that prefix and its `build_release` and `build` directories:

```bash
cmake -S full_runs/HW_N512 -B full_runs/HW_N512/build_reproduce \
  -DCMAKE_BUILD_TYPE=Release \
  -DPython_EXECUTABLE="$(command -v python)"
cmake --build full_runs/HW_N512/build_reproduce --parallel 4
```

For another BOUT++ installation, pass `-DBOUT_ROOT=/absolute/path/to/BOUT++`
to the configure command. This accepts an installation prefix, checkout with
`build`/`build_release`, or configured build directory. For a differently named
build directory, you can instead set
`-Dbout++_DIR=/absolute/path/to/directory/containing/bout++Config.cmake`.
Use a fresh build directory when changing compilers or Python installations;
existing build caches and executables contain machine-specific paths.

CMake selects the Python interpreter, development library and NumPy headers
for embedding. Point the callback at packages for **that same Python version**:

```bash
export PBOUT_UNET_SITE_PACKAGES="$(python -c 'import sysconfig; print(sysconfig.get_path("purelib"))')"
export PBOUT_UNET_DEVICE=cpu           # use cuda on a GPU node
export PBOUT_UNET_IMPLICIT=1          # matches implicitStylES=true in hw.cxx
export PBOUT_UNET_RESCALING=0
export PBOUT_UNET_WRITE_DIAGNOSTICS=2  # save field snapshots for plotting
```

#### Prepare separate LES and DNS runs

The shipped input has a 512² active grid. Create independent run directories
and change the LES mesh to 128², retaining four X guard cells. Use a fresh working
directory for each run: diagnostics use the local `results_U-Net` directory,
and reusing it can overwrite or mix snapshots from different trajectories.

```bash
python - <<'PY'
from pathlib import Path
import re

root = Path('full_runs/HW_N512')
template = (root / 'data/BOUT.inp').read_text()
for name, n in [('les128', 128), ('dns512', 512)]:
    run = root / 'runs' / name / 'data'
    run.mkdir(parents=True, exist_ok=True)
    text = re.sub(r'^nx\s*=.*$', f'nx = {n + 4}', template, flags=re.M)
    text = re.sub(r'^nz\s*=.*$', f'nz = {n}', text, flags=re.M)
    (run / 'BOUT.inp').write_text(text)
PY
```

Run the U-Net closure on the coarse grid, then the DNS baseline from the same
initial snapshot. The examples use a **single MPI rank**, matching the full
plane expected by the current Python callback; multi-rank decomposition needs
separate validation.

```bash
(
  cd "$UNET_ROOT/full_runs/HW_N512/runs/les128"
  export PBOUT_UNET_DNS_INIT=0
  export PBOUT_UNET_BOUT_INP="$PWD/data/BOUT.inp"
  "$UNET_ROOT/full_runs/HW_N512/build_reproduce/hasegawa-wakatani" -d data
)
(
  cd "$UNET_ROOT/full_runs/HW_N512/runs/dns512"
  export PBOUT_UNET_DNS_INIT=2
  export PBOUT_UNET_BOUT_INP="$PWD/data/BOUT.inp"
  "$UNET_ROOT/full_runs/HW_N512/build_reproduce/hasegawa-wakatani" -d data
)
```

Mode `0` reconstructs fine-grid fields and supplies filtered nonlinear terms
to the coarse solver. Mode `2` evolves unfiltered DNS without U-Net
reconstruction, using the checkpoint for grid/domain configuration. Each run
writes BOUT++ outputs under `data/` and diagnostic snapshots under
`results_U-Net/fields/`. The template uses `timestep=1`, `nout=100`. A short
smoke run can use smaller `nout` in **both** generated input files; it will not
reproduce the full plotted time interval.

Keep `[hw] closure_start_step=0` so the Python nonlinear-term callback runs
from startup. A positive value delays the callback (and its field diagnostics)
by that many internal solver steps, which would change the closure experiment.

#### Plot and compare trajectories

```bash
python plot_unet_validation.py \
  "$UNET_ROOT/full_runs/HW_N512/runs/les128/results_U-Net/fields" \
  "$UNET_ROOT/full_runs/HW_N512/runs/dns512/results_U-Net/fields" \
  --output-dir "$UNET_ROOT/results/reproduction_HW_N512/comparison" \
  --lx 50.176 --ly 50.176 --spectrum-cutoff-n 128
```

Absolute input paths avoid the plotting script's legacy BOUT++ base-directory
default. The plot destination is recreated on each invocation. Outputs include
`energy_vs_time.png`, `enstrophy_vs_time.png`,
`hw_energy_spectra_initial_final.png`, `U_pdf_vs_time.png`, `V_pdf_vs_time.png`,
`P_pdf_vs_time.png` and `particle_flux_vs_time.png`. PDFs compare initial and
final spatial distributions. For additional coarse resolutions, repeat the
LES run with its matching checkpoint and mesh, and pass up to two additional
snapshot directories to the plotting script.

### Option 2: train, check and integrate

#### Train on DNS snapshots

Supply the full DNS training dataset for the selected case as `.npz` files,
with spatial dimensions matching `fine_n`. A few initial-condition snapshots
in `fields/` are not the training corpus. The repository does not provide a
verified download location for that corpus; use the original DNS dataset or
generate one with matching physical parameters. Exact reproduction requires
the original data, splits, checkpoint settings and solver configuration.

The following settings match the saved
`training/training_HW_dns512_les128/training_parameters.json` configuration
for the example run (apart from input/output paths and device availability):

```bash
export DNS_DATASET="/absolute/path/to/HW_N512/fields"
export TRAIN_RUN="$UNET_ROOT/training/reproduction_HW_dns512_les128"
python train_germano_unet.py \
  --dataset "$DNS_DATASET" --output-dir "$TRAIN_RUN" \
  --fine-n 512 --coarse-n 128 --lx 50.176 --ly 50.176 --alpha 0.392 \
  --epochs 100 --batch-size 4 --base-channels 24 \
  --precision float64 --no-normalize --no-variational \
  --learning-rate 0.0001 --lambda-mse 0.1 --lambda-vorticity 0.1 \
  --split-mode random --seed 7 --validation-fraction 0.1 --test-fraction 0.1 \
  --inverse-filter cutoff --inverse-cutoff-fraction 0.5 --inverse-cutoff-order 8 \
  --inverse-method van_cittert --van-cittert-iterations 5 \
  --van-cittert-relaxation 1 --post-filter none
```

Random splitting matches this saved run; it does not imply temporal separation
between training and validation. Use explicit `train/`, `validation/` and
`test/` datasets or a sequential split when that separation is required.
`run.sh` is also available as a site-specific SLURM example; adapt its resource
and environment settings before submitting it.

#### Check training with `plot_tensorboard.py`

```bash
python plot_tensorboard.py "$TRAIN_RUN" --output-dir "$TRAIN_RUN/tensorboard_plots"
tensorboard --logdir "$TRAIN_RUN/tensorboard"
```

Inspect training versus validation losses and DNS relative-L2 diagnostics for
convergence and a growing generalization gap. The PNG exporter creates
`tensorboard_summary.png`. This example was generated with that script from
the archived HW 512² → 128² run's TensorBoard events:

![HW 512 to 128 training and validation diagnostics](diagrams/reproduction/tensorboard_summary.png)

#### Check standalone inference, then integrate

```bash
export PBOUT_UNET_CHECKPOINT="$TRAIN_RUN/checkpoints/best.pt"
export PBOUT_UNET_DNS_PATH="$UNET_ROOT/fields/fields_HW_run0_time381_N512.npz"
python infer_germano_unet.py \
  --checkpoint "$PBOUT_UNET_CHECKPOINT" \
  --input "$PBOUT_UNET_DNS_PATH" \
  --output-dir "$UNET_ROOT/inference_output/reproduction_HW_N512"
```

Inspect the saved field/error images, spectra and reconstructed snapshot before
starting a trajectory. The inference output directory is recreated each time.
Then follow **Compile the coupled model**, **Prepare separate LES and DNS runs**
and **Plot and compare trajectories** in Option 1, keeping the newly trained
`PBOUT_UNET_CHECKPOINT`. Snapshot accuracy alone does not establish stable or
accurate time integration.

### Example validation results

These are archived results, not outputs generated by the commands during
README preparation. Legends give the coarse resolutions; the worked example
above reproduces one such resolution at a time. Numerical agreement depends
on the same initial condition, checkpoint and runtime settings.

![HW N512 energy histories](papers/CPC26/images/HW_N512_energy_vs_time.png)

HW N512: the coarsest trajectory has a persistent energy deficit, while finer
reconstructions remain closer to DNS but miss part of its late increase.

![mHW N256 enstrophy histories](papers/CPC26/images/mHW_N256_enstrophy_vs_time.png)

mHW N256: the reconstructed enstrophy burst is delayed and the late-time level
is higher than DNS. This case requires its own mHW checkpoint and physical
configuration; the HW N512 example is not interchangeable with it.

![HW N512 radial and poloidal particle flux](papers/CPC26/images/HW_N512_particle_flux_vs_time.png)

HW N512: radial-flux magnitudes are similar, with larger discrepancies in the
coarsest reconstruction and in the timing of poloidal fluctuations. The paper
in [`papers/CPC26`](papers/CPC26/Paper_Germano_GenerativeAI.tex) includes the
remaining spectra and PDF comparisons.

## Direct coarse-grid U-Net

Training uses `DirectUNetSR`. The encoder now begins directly on the LES grid:
the default 3-channel 64x64 input is mapped to 96 feature channels at 64x64,
without initial bicubic upsampling. A 192-channel 32x32 bottleneck feeds a
periodic bilinear decoder, which returns through 64x64, 128x128, and 256x256.
The model predicts the complete filtered fields and has no global input skip.

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

The default decoder uses periodic bilinear interpolation followed by learned
convolutions, without the added fixed smoothing filters.

Checkpoints record `architecture: DirectUNetSR`. Existing `FilteredDirectUNetSR`
checkpoints remain supported and retain their filtered decoder when loaded.
Earlier `LegacyResidualUNetSR`, `BicubicUNetSR`, `ResidualUNetSR`,
`PeriodicResidualCNN`, and `PeriodicSubpixelCNN` checkpoints remain supported.
The small bicubic CNN implementation has been removed; its saved training
results remain available, but its checkpoints cannot be loaded by this code.

## Training reference

`train_germano_unet.py` recursively reads `.npz` DNS snapshots from the
dataset directory. A dataset may either contain all snapshots in one tree, in
which case the script creates the train/validation/test split, or contain
explicit `train/`, `validation/`, and `test/` subdirectories.

See Option 2 above for a complete training command matching the archived HW 512² → 128² example.

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
| `1` | Write all field, image, Arakawa, and animation diagnostics. |
| `2` | Write only field snapshots under `results_U-Net/fields` (default). |

Diagnostic mode `1` recreates `results_U-Net` during initialization. Use a fresh
working directory for each run in every mode to avoid overwriting or mixing
diagnostic snapshots.

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

## PBOUT environment-variable defaults

The following values are used when the corresponding environment variable is
not set. A checkpoint/training fallback means that the value embedded in the
selected checkpoint takes precedence over the final fallback shown here.

| Environment variable | Default and available options |
| --- | --- |
| `PBOUT_UNET_ALPHA` | **Default:** checkpoint `alpha`.<br>**Options:** any finite number greater than or equal to `0`. |
| `PBOUT_UNET_BOUT_INP` | **Default:** `data/BOUT.inp`.<br>**Options:** path to the BOUT++ input file used by the run. |
| `PBOUT_UNET_CHECKPOINT` | **Default:** `<repository>/results/germano_training_output_v10/checkpoints/best.pt`.<br>**Options:** path to a compatible `.pt` checkpoint. |
| `PBOUT_UNET_DEVICE` | **Default:** `cuda` when available; otherwise `cpu`.<br>**Options:** a valid PyTorch device, normally `cpu`, `cuda`, or `cuda:N`. |
| `PBOUT_UNET_DIR` | **Default:** none; it is not read directly by `pBOUT-unet.py`.<br>**Options:** repository path for launch-script use. |
| `PBOUT_UNET_DNS_FIELD_DIR` | **Default:** `results_U-Net_DNS/fields`.<br>**Options:** directory containing mode-3 DNS replay `.npz` snapshots. |
| `PBOUT_UNET_DNS_INIT` | **Default:** `0`.<br>**Options:**<br>`0` = initialize with Germano-filtered, downsampled LES fields from `PBOUT_UNET_DNS_PATH`.<br>`1` = initialize with filtered DNS and reconstruct according to `PBOUT_UNET_RECONSTRUCT_CHANNELS`.<br>`2` = initialize and run with unfiltered DNS fields and terms.<br>`3` = initialize like mode `0`, then replay snapshots from `PBOUT_UNET_DNS_FIELD_DIR`. |
| `PBOUT_UNET_DNS_PATH` | **Default:** unset (`None`).<br>**Options:** path to the initial DNS `.npz` snapshot. |
| `PBOUT_UNET_IMPLICIT` | **Default:** `1`.<br>**Options:** `0` = return DNS brackets; `1` = return incoming BOUT++ brackets minus DNS brackets. |
| `PBOUT_UNET_INVERSE_CUTOFF_FRACTION` | **Default:** checkpoint value; otherwise `None`.<br>**Options:** float in `(0, 1]`; `none`, `off`, `disable`, or `disabled` disables it. |
| `PBOUT_UNET_INVERSE_CUTOFF_ORDER` | **Default:** checkpoint value; otherwise `8.0`.<br>**Options:** any finite positive number. |
| `PBOUT_UNET_INVERSE_FILTER` | **Default:** checkpoint/training value; otherwise `cutoff`.<br>**Options:** `cutoff` or `filtered_dns_spectrum`; hyphens are normalized to underscores. |
| `PBOUT_UNET_INVERSE_METHOD` | **Default:** checkpoint/training value; otherwise `exact`.<br>**Options:** `exact` or `van_cittert`; hyphens are normalized to underscores. |
| `PBOUT_UNET_INVERSE_MIN_WAVENUMBER` | **Default:** checkpoint value; otherwise `1.0`.<br>**Options:** any finite number greater than or equal to `0`. |
| `PBOUT_UNET_LAPLACIAN_ORDER` | **Default:** checkpoint/training value; otherwise `2`.<br>**Options:** `2` or `4`. |
| `PBOUT_UNET_OMEGA_FROM_POTENTIAL` | **Default:** derived from checkpoint vorticity mode.<br>**Options:** `0`, `false`, `no`, and `off` are false; any other value is true. A Laplacian-output checkpoint forces true. |
| `PBOUT_UNET_POST_CUTOFF_FRACTION` | **Default:** checkpoint value; otherwise `None`.<br>**Options:** float in `(0, 1]`; `auto` uses `coarse_n/fine_n`; `none`, `off`, `disable`, or `disabled` disables it. |
| `PBOUT_UNET_POST_CUTOFF_ORDER` | **Default:** checkpoint value; otherwise `8.0`.<br>**Options:** any finite positive number. |
| `PBOUT_UNET_POST_FILTER` | **Default:** checkpoint/training value; otherwise `none`.<br>**Options:** `none`, `high_order`, or `gaussian`; hyphens are normalized to underscores. |
| `PBOUT_UNET_RECONSTRUCT_CHANNELS` | **Default:** `none`.<br>**Options:** `none`/`off`, `all`, or a comma-separated subset of checkpoint field names, such as `U` or `U,V`. |
| `PBOUT_UNET_RESCALING` | **Default:** `0`.<br>**Options:** `0` = off; `1` = formula 11; `2` = formula 13; `3` = initial-DNS maxima. |
| `PBOUT_UNET_SITE_PACKAGES` | **Default:** Python-version-dependent; see below.<br>**Options:** matching site-packages path or an empty string to retain the interpreter's existing paths. |
| `PBOUT_UNET_VAN_CITTERT_ITERATIONS` | **Default:** checkpoint/training value; otherwise `1`.<br>**Options:** any integer greater than or equal to `0`. |
| `PBOUT_UNET_VAN_CITTERT_RELAXATION` | **Default:** checkpoint/training value; otherwise `1.0`.<br>**Options:** float in `(0, 1]`. |
| `PBOUT_UNET_WRITE_DIAGNOSTICS` | **Default:** `2`.<br>**Options:** `0` = off; `1` = all diagnostics; `2` = field snapshots only. |

`PBOUT_UNET_SITE_PACKAGES` has these interpreter-specific defaults:

- Python 3.11: `/gpfs/mcc/HT04543/jxc06/jxc74-jxc06/programs/miniconda3/envs/py311/lib/python3.11/site-packages`
- Python 3.13: `~/.pyenvs/BOUT-Hermes3/lib/python3.13/site-packages`
- Other Python versions: an empty string, which uses the interpreter's existing
  package paths

The default checkpoint path must exist for checkpoint-derived defaults to be
available. If it does not, set `PBOUT_UNET_CHECKPOINT` to a valid checkpoint.
