#!/bin/bash

#SBATCH --job-name=test
#SBATCH --output=out.txt
#SBATCH --error=err.txt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

export CUDA_VISIBLE_DEVICES=0

source ~/downloads/spack/share/spack/setup-env.sh
spack env activate BOUT-Hermes3
source ~/.pyenvs/BOUT-Hermes3/bin/activate

# Set these when submitting to choose the grids independently, for example:
#   DNS_N=256 LES_N=64 sbatch run.sh
#   DNS_N=512 LES_N=128 sbatch run.sh
DNS_N="${DNS_N:-256}"
LES_N="${LES_N:-64}"
LX="${LX:-50.176}"

ALPHA="${ALPHA:-$(awk -v lx="${LX}" -v n="${LES_N}" 'BEGIN {print lx / n}')}"

CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-25}"
DIAGNOSTIC_EVERY="${DIAGNOSTIC_EVERY:-5}"
DATASET="${DATASET:-../../data/BOUT_runs/HW_2D/Papers/PoP23/HW_N${DNS_N}/fields/}"
OUTPUT_DIR="${OUTPUT_DIR:-training_dns${DNS_N}_les${LES_N}}"

# Select the pre-inversion filter, for example:
#   INVERSE_FILTER=cutoff sbatch run.sh
# The historical resolution-dependent cutoff fraction is a fraction of Nyquist:
# DNS 256 with LES 32/64/128 gives 0.44/0.5/0.707107 respectively.
INVERSE_FILTER="${INVERSE_FILTER:-cutoff}"

python train_germano_unet.py \
  --dataset "${DATASET}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs 100 \
  --batch-size 4 \
  --fine-n "${DNS_N}" \
  --coarse-n "${LES_N}" \
  --alpha "${ALPHA}" \
  --precision float64 \
  --lx "${LX}" \
  --ly 50.176 \
  --lambda-vorticity 0.1 \
  --save-every "${CHECKPOINT_EVERY}" \
  --diagnostic-every "${DIAGNOSTIC_EVERY}" \
  --post-filter none \
  --no-normalize \
  --inverse-filter "${INVERSE_FILTER}" \
  --inverse-cutoff-fraction 0.707107  \
  --inverse-min-wavenumber 1.0 \
  --no-variational --lambda-kl 0.0 \
  --inverse-method van_cittert \
  --van-cittert-iterations 5 \
  --van-cittert-relaxation 1.0

#python infer_germano_unet.py --checkpoint germano_training_output_32_v3/checkpoints/best.pt --input fields_HW_run0_time501.npz --output-dir inference_output
