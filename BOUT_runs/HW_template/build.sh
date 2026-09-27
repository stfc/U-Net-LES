#!/usr/bin/env bash
# Usage: bash BOUT_runs/HW_template/build.sh BOUT_runs/HW_N256 [extra CMake arguments]
# Copy data/BOUT.inp unchanged; all simulation parameters come from that file.
set -euo pipefail
template_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ $# -eq 0 || -z "$1" || "$1" == -* ]]; then
    echo "Usage: bash BOUT_runs/HW_template/build.sh RUN_DIRECTORY [extra CMake arguments]" >&2
    echo "Example: bash BOUT_runs/HW_template/build.sh BOUT_runs/HW_N256" >&2
    exit 1
fi
run_dir="$(realpath -m -- "$1")"
shift
if [[ "$(dirname -- "$run_dir")" != "$(dirname -- "$template_dir")" || "$run_dir" == "$template_dir" ]]; then
    echo "RUN_DIRECTORY must be a direct child of BOUT_runs, other than HW_template" >&2
    exit 1
fi
bout_root="${BOUT_ROOT:-$(cd -- "$template_dir/../../../BOUT-dev" && pwd)}"

# Use the cluster environment that supplies BOUT-dev's dependencies.
if [[ -f ~/downloads/spack/share/spack/setup-env.sh ]]; then
    source ~/downloads/spack/share/spack/setup-env.sh
    spack env activate BOUT-Hermes3
fi
if [[ -f ~/.pyenvs/BOUT-Hermes3/bin/activate ]]; then
    source ~/.pyenvs/BOUT-Hermes3/bin/activate
fi

mkdir -p "$run_dir/data"
if [[ ! -e "$run_dir/data/BOUT.inp" ]]; then
    cp -- "$template_dir/data/BOUT.inp" "$run_dir/data/BOUT.inp"
fi

# Match the compilers used by BOUT-dev (caller arguments can override these).
compiler_args=()
for cache in "$bout_root/CMakeCache.txt" "$bout_root/build_release/CMakeCache.txt" "$bout_root/build/CMakeCache.txt"; do
    if [[ -f "$cache" ]]; then
        for language in C CXX; do
            compiler="$(sed -n "s/^CMAKE_${language}_COMPILER:[^=]*=//p" "$cache")"
            if [[ -n "$compiler" ]]; then
                compiler_args+=("-DCMAKE_${language}_COMPILER=$compiler")
            fi
        done
        break
    fi
done
cmake -S "$template_dir" -B "$run_dir/build_release" \
    -U 'bout++_DIR' -DBOUT_ROOT="$bout_root" -DCMAKE_BUILD_TYPE=Release \
    -DPython_EXECUTABLE="$(command -v python)" "${compiler_args[@]}" "$@"
cmake --build "$run_dir/build_release" --parallel "${BUILD_JOBS:-4}"

printf 'Ready, run as ./build_release/hasegawa-wakatani'