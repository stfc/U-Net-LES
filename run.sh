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
  --precision float64 \
  --lx "${LX}" \
  --ly 50.176 \
  --lambda-spectrum 1e6 \
  --lambda-dns 0.1 \
  --save-every 5 \
  --no-normalize

#python infer_germano_unet.py --checkpoint germano_training_output_32_v3/checkpoints/best.pt --input fields_HW_run0_time501.npz --output-dir inference_output
