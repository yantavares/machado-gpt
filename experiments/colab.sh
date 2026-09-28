#!/usr/bin/env bash
# Roda os experimentos numa VM do Colab com GPU via google-colab-cli.
# Autoria: código original do aluno.
#
#   experiments/colab.sh novo            # cria a sessão T4, envia o projeto, instala deps
#   experiments/colab.sh lanca [runs]    # dispara executar.py em segundo plano (padrão: --todos)
#   experiments/colab.sh status          # status.json + fim do log
#   experiments/colab.sh baixa           # traz results/ e os checkpoints enxutos
#   experiments/colab.sh para            # libera a VM (sempre rodar ao terminar)
set -euo pipefail
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
COLAB="${COLAB:-$HOME/anaconda3/envs/ai/bin/colab}"
S="${SESSAO:-machado}"
REMOTO=/content/proj
PACOTE="${TMPDIR:-/tmp}/machado_proj.tar.gz"

remoto() { # executa python na VM
    echo "$1" | "$COLAB" exec -s "$S" --timeout "${2:-120}"
}

case "${1:-}" in
novo)
    tar -czf "$PACOTE" -C "$RAIZ" \
        machado_gpt experiments tests \
        nanoGPT-master/model.py nanoGPT-master/train.py nanoGPT-master/configurator.py \
        nanoGPT-master/sample.py nanoGPT-master/data/machado_char nanoGPT-master/data/machado_gpt2 \
        nanoGPT-master/data/machado_ptbr \
        corpus/train.txt corpus/val.txt corpus/test.txt corpus/test_por_genero
    ls -lh "$PACOTE"
    "$COLAB" new -s "$S" --gpu T4
    "$COLAB" upload -s "$S" "$PACOTE" /content/machado_proj.tar.gz
    remoto "import subprocess as sp
print(sp.run('mkdir -p $REMOTO && tar -xzf /content/machado_proj.tar.gz -C $REMOTO && pip install -q tiktoken transformers && ls $REMOTO && nvidia-smi --query-gpu=name,memory.total --format=csv', shell=True, capture_output=True, text=True, errors='replace'))" 600
    ;;
lanca)
    shift
    ARGS="${*:---todos}"
    remoto "import subprocess as sp
sp.Popen('cd $REMOTO && nohup python experiments/executar.py $ARGS >> results_executar.log 2>&1 &', shell=True)
print('lançado: $ARGS')"
    ;;
status)
    remoto "import subprocess as sp
print(sp.run('cat $REMOTO/results/status.json 2>/dev/null | tail -40; echo; tail -c 1500 $REMOTO/results_executar.log; echo; for f in $REMOTO/runs/*/train.log; do echo \$f; grep -E \"^step|^iter\" \$f | tail -2; done; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv', shell=True, capture_output=True, text=True, errors='replace').stdout)"
    ;;
baixa)
    remoto "import subprocess as sp
print(sp.run('cd $REMOTO && tar -czf /content/resultados.tar.gz results results_executar.log runs/*/train.log runs/*/treino.json', shell=True, capture_output=True, text=True, errors='replace'))" 300
    "$COLAB" download -s "$S" /content/resultados.tar.gz "$RAIZ/resultados_colab.tar.gz"
    tar -xzf "$RAIZ/resultados_colab.tar.gz" -C "$RAIZ" && rm "$RAIZ/resultados_colab.tar.gz"
    if [ "${2:-}" = "--ckpt" ]; then
        for r in $(remoto "import glob; print(' '.join(p.split('/')[-2] for p in glob.glob('$REMOTO/runs/*/ckpt_slim.pt')))"); do
            mkdir -p "$RAIZ/runs/$r"
            "$COLAB" download -s "$S" "$REMOTO/runs/$r/ckpt_slim.pt" "$RAIZ/runs/$r/ckpt_slim.pt"
        done
    fi
    ;;
para)
    "$COLAB" stop -s "$S"
    ;;
*)
    sed -n '2,11p' "$0"; exit 1 ;;
esac
