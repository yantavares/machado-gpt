#!/usr/bin/env bash
# Roda um comando com teto de memória e de threads, para não travar a máquina.
# Autoria: código original do aluno.
#
#   experiments/limitado.sh python -m machado_gpt.geracao --run runs/ptbr_ft --prompt "..."
#   MEMORIA=4G THREADS=4 experiments/limitado.sh python ...
set -euo pipefail
export OMP_NUM_THREADS="${THREADS:-6}" MKL_NUM_THREADS="${THREADS:-6}"
if command -v systemd-run >/dev/null 2>&1; then
    exec systemd-run --user --scope -q -p MemoryMax="${MEMORIA:-6G}" -p MemorySwapMax=0 nice -n 19 "$@"
else
    exec nice -n 19 "$@"   # sem systemd (ex.: macOS): só limita as threads
fi
