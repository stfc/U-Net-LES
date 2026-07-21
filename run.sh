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

python train_germano_unet.py --output-dir germano_training_output_32_v3 --dataset ../../data/BOUT_runs/HW_2D/Papers/PoP23/HW_N256/fields/ --epochs 300 --batch-size 4 --alpha-factor 10.0 --coarse-n 32 --learn-vorticity
