"""Geração de amostras e métricas de qualidade do texto gerado.

Autoria: código original do aluno. A amostragem usa `GPT.generate` do nanoGPT
(Karpathy, MIT), com temperatura 0,8 e top-k 200, os padrões de `sample.py`.

Protocolo: 10 parágrafos longos sorteados (semente fixa) do conjunto de teste
viram prompts (os primeiros ~150 caracteres, cortados em fim de palavra). Cada
modelo gera 3 continuações de 400 caracteres por prompt. A continuação real do
parágrafo serve de referência humana para as mesmas métricas:

- palavras válidas: fração das palavras geradas que existem no vocabulário de
  treino (mede ortografia; nomes próprios novos contam como inválidos);
- distinct-1/2 (Li et al., 2016): n-gramas distintos / total, diversidade;
- memorização: fração dos 8-gramas de palavras gerados que aparecem literalmente
  no treino (cópia do corpus em vez de texto novo);
- repetição: fração de 4-gramas repetidos dentro da própria amostra.

Uso:
    python -m machado_gpt.geracao --run runs/char_baby
    python -m machado_gpt.geracao --referencia        # métricas do texto humano
    python -m machado_gpt.geracao --run runs/ptbr_ft --prompt "Capitu olhou para mim"
"""

import argparse
import json
import math
import random
import re
from pathlib import Path

import torch

from machado_gpt import tokenizacao
from machado_gpt.avaliacao import NANOGPT, CORPUS, RAIZ, carrega_hf, carrega_run

N_PROMPTS, AMOSTRAS_POR_PROMPT, CHARS_PROMPT, CHARS_GERADOS = 10, 3, 150, 400
SEMENTE = 1337

# prompts de vitrine para o artigo e a apresentação (não estão no corpus)
VITRINE = [
    "Capitu olhou para mim",
    "— Não, senhor, disse ele;",
    "A vida é uma ópera,",
]


def palavras(texto):
    return re.findall(r"[^\W\d_]+", texto.lower())


def prompts_do_teste(n=N_PROMPTS, semente=SEMENTE):
    pars = [p for p in (CORPUS / "test.txt").read_text(encoding="utf-8").split("\n")
            if len(p) >= CHARS_PROMPT + CHARS_GERADOS + 50]
    rng = random.Random(semente)
    escolhidos = rng.sample(pars, n)
    saida = []
    for p in escolhidos:
        corte = p.rfind(" ", 0, CHARS_PROMPT)
        saida.append({"prompt": p[:corte], "referencia": p[corte:corte + CHARS_GERADOS]})
    return saida


class Metricas:
    """Métricas de texto; carrega o vocabulário e os 8-gramas do treino uma vez."""

    def __init__(self):
        treino = (CORPUS / "train.txt").read_text(encoding="utf-8")
        ws = palavras(treino)
        self.vocab = set(ws)
        self.oito = {hash(tuple(ws[i:i + 8])) for i in range(len(ws) - 7)}

    def calcula(self, textos):
        todas, uni, bi, oito_tot, oito_mem, rep_tot, rep = [], set(), set(), 0, 0, 0, 0
        n_bi = 0
        for t in textos:
            ws = palavras(t)
            todas.extend(ws)
            uni.update(ws)
            bigr = [tuple(ws[i:i + 2]) for i in range(len(ws) - 1)]
            bi.update(bigr)
            n_bi += len(bigr)
            g8 = [hash(tuple(ws[i:i + 8])) for i in range(len(ws) - 7)]
            oito_tot += len(g8)
            oito_mem += sum(h in self.oito for h in g8)
            g4 = [tuple(ws[i:i + 4]) for i in range(len(ws) - 3)]
            rep_tot += len(g4)
            rep += len(g4) - len(set(g4))
        return {
            "palavras": len(todas),
            "palavras_validas": sum(w in self.vocab for w in todas) / max(1, len(todas)),
            "distinct_1": len(uni) / max(1, len(todas)),
            "distinct_2": len(bi) / max(1, n_bi),
            "memorizacao_8grama": oito_mem / max(1, oito_tot),
            "repeticao_4grama": rep / max(1, rep_tot),
        }


def gera_texto(modelo, tok, prompt, n_chars, device, ratio):
    ids = torch.tensor([tok.encode(prompt)], dtype=torch.long, device=device)
    # tokens suficientes para passar de n_chars; o excesso é cortado
    n_tokens = math.ceil(n_chars / ratio * 1.3) + 8
    ctx = (torch.amp.autocast(device_type="cuda", dtype=torch.float16)
           if "cuda" in str(device) else torch.no_grad())
    with ctx:
        y = modelo.gera(ids, n_tokens)
    texto = tok.decode(y[0].tolist())
    return texto[len(prompt):][:n_chars]


def roda(modelo, device, saida_dir, n_prompts=N_PROMPTS, por_prompt=AMOSTRAS_POR_PROMPT,
         n_chars=CHARS_GERADOS):
    data_dir = NANOGPT / "data" / modelo.dataset
    tok = tokenizacao.carrega(data_dir)
    info = json.loads((data_dir / "info.json").read_text())
    ratio = info["train"]["caracteres_por_token"]
    torch.manual_seed(SEMENTE)

    amostras = []
    for p in prompts_do_teste()[:n_prompts]:
        for k in range(por_prompt):
            amostras.append({"prompt": p["prompt"],
                             "gerado": gera_texto(modelo, tok, p["prompt"], n_chars, device, ratio)})
    vitrine = [{"prompt": v, "gerado": gera_texto(modelo, tok, v, int(1.5 * n_chars), device, ratio)}
               for v in VITRINE]

    metricas = Metricas().calcula([a["gerado"] for a in amostras])
    saida_dir = Path(saida_dir)
    saida_dir.mkdir(parents=True, exist_ok=True)
    (saida_dir / f"{modelo.nome.replace('/', '_')}.json").write_text(json.dumps(
        {"modelo": modelo.nome, "metricas": metricas, "amostras": amostras, "vitrine": vitrine},
        ensure_ascii=False, indent=2), encoding="utf-8")
    return metricas


def referencia(saida_dir):
    ps = prompts_do_teste()
    m = Metricas().calcula([p["referencia"] for p in ps])
    Path(saida_dir).mkdir(parents=True, exist_ok=True)
    (Path(saida_dir) / "referencia_humana.json").write_text(json.dumps(
        {"modelo": "referência humana (teste)", "metricas": m, "amostras": ps},
        ensure_ascii=False, indent=2), encoding="utf-8")
    return m


def main():
    ap = argparse.ArgumentParser(description="Gera amostras e mede a qualidade do texto.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run")
    g.add_argument("--hf")
    g.add_argument("--referencia", action="store_true")
    ap.add_argument("--prompt", help="gera só a continuação deste texto (uso livre do modelo)")
    ap.add_argument("--chars", type=int, default=600, help="tamanho da continuação com --prompt")
    ap.add_argument("--dataset")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--saida", default=str(RAIZ / "results" / "amostras"))
    args = ap.parse_args()
    if args.prompt:
        modelo = (carrega_run(args.run, args.device) if args.run
                  else carrega_hf(args.hf, args.dataset, 1024, args.device))
        data_dir = NANOGPT / "data" / modelo.dataset
        tok = tokenizacao.carrega(data_dir)
        ratio = json.loads((data_dir / "info.json").read_text())["train"]["caracteres_por_token"]
        torch.manual_seed(SEMENTE)
        print(args.prompt + gera_texto(modelo, tok, args.prompt, args.chars, args.device, ratio))
        return
    if args.referencia:
        m = referencia(args.saida)
    else:
        modelo = (carrega_run(args.run, args.device) if args.run
                  else carrega_hf(args.hf, args.dataset, 1024, args.device))
        m = roda(modelo, args.device, args.saida)
    print(json.dumps(m, indent=2))


if __name__ == "__main__":
    main()
