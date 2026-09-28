"""Linhas de base de n-gramas de caracteres (unigrama, bigrama, trigrama).

Autoria: código original do aluno.

O primeiro modelo do vídeo "Let's build GPT" é um bigrama neural: uma tabela
de embeddings V x V treinada por gradiente. No ótimo, ele coincide com o
bigrama de contagens com suavização, que é calculado aqui em forma fechada.
Isso dá o piso de referência sem ruído de otimização. A suavização aditiva
(alfa) é escolhida na validação.

Uso:
    python -m machado_gpt.baselines
"""

import json
import math
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
DATA = RAIZ / "nanoGPT-master" / "data" / "machado_char"
ALFAS = [1e-3, 1e-2, 0.1, 0.5, 1.0]


def contagens(ids, n, V):
    tab = np.zeros((V,) * n, dtype=np.float64)
    idx = tuple(ids[i:len(ids) - n + 1 + i] for i in range(n))
    np.add.at(tab, idx, 1)
    return tab


def bpc(tab, ids, n, alfa):
    V = tab.shape[-1]
    probs = (tab + alfa) / (tab.sum(axis=-1, keepdims=True) + alfa * V)
    idx = tuple(ids[i:len(ids) - n + 1 + i] for i in range(n))
    return float(-np.log2(probs[idx]).mean())


def main():
    V = json.loads((DATA / "info.json").read_text())["vocab_size"]
    ids = {s: np.fromfile(DATA / f"{s}.bin", dtype=np.uint16).astype(np.int64)
           for s in ("train", "val", "test")}
    res = {}
    for n, nome in [(1, "unigrama"), (2, "bigrama"), (3, "trigrama")]:
        tab = contagens(ids["train"], n, V)
        alfa = min(ALFAS, key=lambda a: bpc(tab, ids["val"], n, a))
        res[nome] = {
            "alfa": alfa,
            "parametros": V ** n,
            "val": {"bpc": bpc(tab, ids["val"], n, alfa)},
            "test": {"bpc": bpc(tab, ids["test"], n, alfa)},
        }
        # perda por token em nats, na mesma escala do train.py
        res[nome]["test"]["perda_por_token"] = res[nome]["test"]["bpc"] * math.log(2)
        print(f"{nome}: alfa={alfa} val={res[nome]['val']['bpc']:.4f} "
              f"test={res[nome]['test']['bpc']:.4f} bpc")
    saida = RAIZ / "results" / "metrics" / "baselines_ngrama.json"
    saida.parent.mkdir(parents=True, exist_ok=True)
    saida.write_text(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
