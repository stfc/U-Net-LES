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

python train_germano_unet.py \
  --dataset ../../data/BOUT_runs/HW_2D/Papers/PoP23/HW_N256/fields/ \
  --output-dir germano_training_output_32_v10 \
  --epochs 300 \
  --batch-size 4 \
  --coarse-n 32 \
  --alpha-factor 8.0 \
  --precision float64 \
  --lx 50.176 \
  --ly 50.176 \
  --lambda-spectrum 10.0 \
  --lambda-laplacian 1 \
  --no-normalize \
  --post-cutoff-fraction 0.4 \
  --invert-derived-vorticity

#python infer_germano_unet.py --checkpoint germano_training_output_32_v3/checkpoints/best.pt --input fields_HW_run0_time501.npz --output-dir inference_output
