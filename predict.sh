#!/usr/bin/env bash
# One-click membrane permeability prediction with CHEDDAR.
#
#   ./predict.sh                   a worked example: the three peptides of Figure 5
#   ./predict.sh 5222              one peptide, by its CycPeptMPDB id
#   ./predict.sh 5222 4705 4271    several at once
#
# Anything else is passed straight through, so these also work:
#   ./predict.sh 4705 --split OD_Murcko      use the scaffold-split model
#   ./predict.sh --partition test --out p.csv   a whole test partition to a csv
#
# The first run builds the two ensemble caches from the MD trajectories and
# takes a few minutes. Every run after that takes a few seconds.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v python >/dev/null 2>&1; then
    echo "No 'python' on your PATH. Set the environment up once with:" >&2
    echo "    conda env create -f environment.yml && conda activate cheddar" >&2
    exit 1
fi

if [ $# -eq 0 ]; then
    echo "No peptide named, so here are the three shown in Figure 5 of the paper."
    echo "Give your own with, for example:  ./predict.sh 5222"
    echo
    set -- 5222 4705 4271
fi

exec python -m cheddar predict "$@"
