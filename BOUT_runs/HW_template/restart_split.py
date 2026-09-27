# cd /gpfs/mcc/HT04543/jxc06/jxc74-jxc06/projects/FusionComputingLab/PhaseV_FCL1/codes/BOUT_repo/build_release/examples/hasegawa-wakatani

# source ~/downloads/spack/share/spack/setup-env.sh
# spack env activate BOUT-Hermes3
# source ~/.pyenvs/BOUT-Hermes3/bin/activate

export N=32                       # Desired MPI task count
export NEW_RUN="data_restart_${N}"

python - <<'PY'
import os
import shutil
from pathlib import Path
from boutdata.restart import redistribute

n = int(os.environ["N"])
out = Path(os.environ["NEW_RUN"])

assert n > 0 and 512 % n == 0 and 512 // n >= 2
out.mkdir()  # Refuses to overwrite an existing directory
shutil.copy2("data/BOUT.inp", out / "BOUT.inp")

ok = redistribute(
    n,
    path="data",
    output=str(out),
    nxpe=n,
    informat="nc",
    outformat="nc",
    mxg=2,
    myg=0,
)
if not ok:
    raise RuntimeError("Restart redistribution failed")
print(f"Restart files ready in {out}")
PY

