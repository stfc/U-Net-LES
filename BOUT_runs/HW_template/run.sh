#!/bin/bash

#SBATCH --job-name=test
#SBATCH --output=out.txt
#SBATCH --error=err.txt
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32

set -euo pipefail
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"

source ~/downloads/spack/share/spack/setup-env.sh
spack env activate BOUT-Hermes3
source ~/.pyenvs/BOUT-Hermes3/bin/activate

if [[ "${1:-}" == "--combine" ]]; then
    shift
    exec python "$@"
fi
executable=./build_release/hasegawa-wakatani
[[ -x "$executable" ]] || executable=./hasegawa-wakatani
mpirun -n "${SLURM_NTASKS:-1}" "$executable" "$@"
