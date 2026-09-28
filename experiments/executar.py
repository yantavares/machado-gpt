"""Orquestra os experimentos: treino -> avaliação -> geração -> checkpoint enxuto.

Autoria: código original do aluno. O treino dos modelos nanoGPT chama o
`train.py` do nanoGPT (Karpathy) como subprocesso, sem alteração de fluxo.

Cada etapa é pulada se a saída já existe, então o script pode ser relançado
depois de uma queda da sessão do Colab. O andamento vai para
`results/status.json`.

Uso:
    python experiments/executar.py char_baby video_gpt ...   # na GPU
    python experiments/executar.py --todos
    python experiments/executar.py char_cpu --device cpu      # local
    python experiments/executar.py char_baby --smoke          # teste rápido na CPU
"""

import argparse
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ))
NANOGPT = RAIZ / "nanoGPT-master"
CONFIGS = RAIZ / "experiments" / "configs"

HF_PTBR = "pierreguillou/gpt2-small-portuguese"
RUNS = {
    # nome: como treinar
    "char_cpu": {"tipo": "nanogpt", "config": "char_cpu.py"},
    "video_gpt": {"tipo": "video"},
    "char_baby": {"tipo": "nanogpt", "config": "char_baby.py"},
    "char_large": {"tipo": "nanogpt", "config": "char_large.py"},
    "gpt2_scratch": {"tipo": "nanogpt", "config": "gpt2_scratch.py"},
    "ptbr_scratch": {"tipo": "nanogpt", "config": "ptbr_scratch.py"},
    "gpt2_zeroshot": {"tipo": "hf", "hf": "gpt2", "dataset": "machado_gpt2", "block": 512},
    "ptbr_zeroshot": {"tipo": "hf", "hf": HF_PTBR, "dataset": "machado_ptbr", "block": 512},
    "gpt2_ft": {"tipo": "nanogpt", "config": "gpt2_ft.py"},
    "ptbr_ft": {"tipo": "nanogpt", "config": "ptbr_ft.py"},
}
ORDEM_GPU = ["char_baby", "ptbr_zeroshot", "gpt2_zeroshot", "ptbr_ft", "gpt2_ft",
             "ptbr_scratch", "gpt2_scratch", "char_large", "video_gpt"]

# sobrescritas do modo --smoke: poucas iterações, modelo pequeno, CPU
SMOKE = ["--device=cpu", "--compile=False", "--dtype=float32", "--max_iters=6", "--lr_decay_iters=6",
         "--warmup_iters=2", "--eval_interval=3", "--eval_iters=2", "--log_interval=1",
         "--batch_size=2", "--gradient_accumulation_steps=1"]
SMOKE_SCRATCH = ["--n_layer=2", "--n_head=2", "--n_embd=64", "--block_size=64"]
SMOKE_FT = ["--block_size=64"]


class Status:
    def __init__(self, raiz_saida):
        self.arq = raiz_saida / "status.json"
        self.dados = json.loads(self.arq.read_text()) if self.arq.exists() else {}

    def marca(self, run, etapa, **extra):
        self.dados.setdefault(run, {})[etapa] = {"quando": time.strftime("%H:%M:%S"), **extra}
        self.arq.parent.mkdir(parents=True, exist_ok=True)
        self.arq.write_text(json.dumps(self.dados, indent=2, ensure_ascii=False))
        print(f"[executar] {run}: {etapa} {extra if extra else ''}", flush=True)


def le_log_nanogpt(log):
    """Extrai as curvas de perda e o tempo por iteração do stdout do train.py."""
    evals = [{"iter": int(a), "train": float(b), "val": float(c)} for a, b, c in
             re.findall(r"step (\d+): train loss ([\d.]+), val loss ([\d.]+)", log)]
    iters = [{"iter": int(a), "loss": float(b), "ms": float(c)} for a, b, c in
             re.findall(r"iter (\d+): loss ([\d.]+), time ([\d.]+)ms", log)]
    # descarta as primeiras iterações (compilação/aquecimento) na mediana
    ms = [i["ms"] for i in iters if i["iter"] >= 20] or [i["ms"] for i in iters]
    return {"evals": evals, "iters": iters, "ms_por_iter_mediana": statistics.median(ms) if ms else None}


def treina(run, spec, run_dir, args, status):
    if (run_dir / "treino.json").exists():
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    status.marca(run, "treino_inicio")
    if spec["tipo"] == "video":
        cmd = [sys.executable, "-m", "machado_gpt.video_gpt", "--out", str(run_dir),
               "--device", args.device]
        if args.smoke:
            cmd += ["--max-iters", "6", "--eval-iters", "2", "--batch-size", "2", "--block-size", "64",
                    "--n-embd", "64", "--n-head", "2", "--n-layer", "2"]
        cwd = RAIZ
    else:
        cmd = [sys.executable, "train.py", str(CONFIGS / spec["config"]), f"--out_dir={run_dir}"]
        if args.smoke:
            cmd += SMOKE + (SMOKE_FT if spec["config"].endswith("_ft.py") else SMOKE_SCRATCH)
        elif args.device == "cpu" and spec["config"] != "char_cpu.py":
            cmd += ["--device=cpu", "--compile=False", "--dtype=float32"]
        cwd = NANOGPT
    with open(run_dir / "train.log", "w") as log:
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}  # log legível durante o treino
        proc = subprocess.run(cmd, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, env=env)
    texto = (run_dir / "train.log").read_text()
    if proc.returncode != 0:
        status.marca(run, "treino_erro", codigo=proc.returncode)
        raise SystemExit(f"{run}: treino falhou\n{texto[-3000:]}")
    if spec["tipo"] == "video":
        resumo = json.loads((run_dir / "treino.json").read_text())
    else:
        resumo = le_log_nanogpt(texto)
    resumo["segundos_total"] = round(time.time() - t0, 1)
    (run_dir / "treino.json").write_text(json.dumps(resumo, indent=2))
    status.marca(run, "treino_fim", segundos=resumo["segundos_total"])


def enxuga(run_dir):
    """Checkpoint só com pesos em fp16 (sem estado do otimizador), para download."""
    import torch
    ck_path = run_dir / "ckpt.pt"
    if not ck_path.exists() or (run_dir / "ckpt_slim.pt").exists():
        return
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    ck.pop("optimizer", None)
    ck["model"] = {k.removeprefix("_orig_mod."): v.half() for k, v in ck["model"].items()}
    torch.save(ck, run_dir / "ckpt_slim.pt")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("runs", nargs="*", help=f"runs a executar: {', '.join(RUNS)}")
    ap.add_argument("--todos", action="store_true", help="todos os runs de GPU, na ordem padrão")
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true", help="teste rápido do pipeline na CPU")
    ap.add_argument("--sem-geracao", action="store_true")
    args = ap.parse_args()

    import torch
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if args.smoke:
        args.device = "cpu"
    runs = ORDEM_GPU if args.todos else args.runs
    desconhecidos = [r for r in runs if r not in RUNS]
    if desconhecidos or not runs:
        ap.error(f"runs inválidos ou ausentes: {desconhecidos}; opções: {list(RUNS)}")
    base_runs = RAIZ / ("runs_smoke" if args.smoke else "runs")
    base_res = RAIZ / ("results_smoke" if args.smoke else "results")
    status = Status(base_res)

    from machado_gpt import avaliacao, geracao

    for run in runs:
        spec = RUNS[run]
        run_dir = base_runs / run
        if spec["tipo"] != "hf":
            treina(run, spec, run_dir, args, status)

        met = base_res / "metrics" / f"{run}.json"
        if not met.exists():
            status.marca(run, "avaliacao_inicio")
            if spec["tipo"] == "hf":
                modelo = avaliacao.carrega_hf(spec["hf"], spec["dataset"], spec["block"], args.device)
            else:
                modelo = avaliacao.carrega_run(run_dir, args.device)
            modelo.nome = run
            res = avaliacao.avalia(modelo, args.device, lote=8 if args.device == "cpu" else 16,
                                   limite_chars=3000 if args.smoke else None)
            if spec["tipo"] != "hf":
                res["treino"] = json.loads((run_dir / "treino.json").read_text())
            met.parent.mkdir(parents=True, exist_ok=True)
            met.write_text(json.dumps(res, indent=2, ensure_ascii=False))
            status.marca(run, "avaliacao_fim", bpc_test=round(res["test"]["bpc"], 4))
        else:
            modelo = None

        amostra = base_res / "amostras" / f"{run}.json"
        if not args.sem_geracao and not amostra.exists():
            if modelo is None:
                modelo = (avaliacao.carrega_hf(spec["hf"], spec["dataset"], spec["block"], args.device)
                          if spec["tipo"] == "hf" else avaliacao.carrega_run(run_dir, args.device))
                modelo.nome = run
            status.marca(run, "geracao_inicio")
            kw = dict(n_prompts=2, por_prompt=1, n_chars=60) if args.smoke else {}
            m = geracao.roda(modelo, args.device, base_res / "amostras", **kw)
            status.marca(run, "geracao_fim", **{k: round(v, 4) for k, v in m.items()})
        if spec["tipo"] != "hf":
            enxuga(run_dir)
        del modelo
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not (base_res / "amostras" / "referencia_humana.json").exists():
        geracao.referencia(base_res / "amostras")
    status.marca("_todos", "fim")


if __name__ == "__main__":
    main()
