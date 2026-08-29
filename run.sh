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

source /gpfs/mcc/HT04543/jxc06/jxc74-jxc06/programs/miniconda3/etc/profile.d/conda.sh
conda activate py311

# Set these when submitting to choose the grids independently, for example:
#   DNS_N=256 LES_N=64 sbatch run.sh
#   DNS_N=512 LES_N=128 sbatch run.sh
DNS_N="${DNS_N:-256}"
LES_N="${LES_N:-64}"
LX="${LX:-50.176}"
ALPHA="${ALPHA:-$(awk -v lx="${LX}" -v n="${LES_N}" 'BEGIN {print lx / n}')}"
# Sublinear grid scaling gives 0.5 and 0.707 for DNS256 LES64/LES128.  A
# 0.44 floor moves the LES32 order-8 roll-off later, where its DNS spectrum
# begins its terminal decay, without changing the two finer cases.
INVERSE_CUTOFF_FRACTION="${INVERSE_CUTOFF_FRACTION:-$(
  awk -v les="${LES_N}" -v dns="${DNS_N}" \
    'BEGIN {fraction = sqrt(les / dns); if (fraction < 0.44) fraction = 0.44; if (fraction > 1.0) fraction = 1.0; print fraction}'
)}"
INVERSE_CUTOFF_ORDER="${INVERSE_CUTOFF_ORDER:-8}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-25}"
DIAGNOSTIC_EVERY="${DIAGNOSTIC_EVERY:-5}"
DATASET="${DATASET:-../../data/BOUT_runs/HW_2D/Papers/PoP23/HW_N${DNS_N}/fields/}"
OUTPUT_DIR="${OUTPUT_DIR:-germano_training_output_dns${DNS_N}_les${LES_N}}"

python train_germano_unet.py \
  --dataset "${DATASET}" \
  --output-dir "${OUTPUT_DIR}" \
  --epochs 100 \
  --batch-size 4 \
  --fine-n "${DNS_N}" \
  --coarse-n "${LES_N}" \
  --alpha "${ALPHA}" \
  --inverse-cutoff-fraction "${INVERSE_CUTOFF_FRACTION}" \
  --inverse-cutoff-order "${INVERSE_CUTOFF_ORDER}" \
  --precision float64 \
  --lx "${LX}" \
  --ly 50.176 \
  --lambda-dns 0.1 \
  --save-every "${CHECKPOINT_EVERY}" \
  --diagnostic-every "${DIAGNOSTIC_EVERY}" \
  --no-normalize \
  --post-filter none \
  --inverse-method van_cittert \
  --van-cittert-iterations 5 \
  --van-cittert-relaxation 1.0

#python infer_germano_unet.py --checkpoint germano_training_output_32_v3/checkpoints/best.pt --input fields_HW_run0_time501.npz --output-dir inference_output
