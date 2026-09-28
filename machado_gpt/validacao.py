"""Confere a cobertura do corpus limpo contra uma segunda fonte independente.

Autoria: código original do aluno.

`archive/obras_machado_de_assis.csv` traz as mesmas 116 obras extraídas de
PDFs (outra edição digital, com quebras de linha na largura da página). Para
cada gênero, mede-se a fração dos 8-gramas de palavras do arquivo alternativo
que também aparecem no corpus limpo. Uma cobertura alta indica que a limpeza
não descartou texto literário; o que falta vem de diferenças de edição
(ortografia, notas, paratexto) e de hifenização dos PDFs.

Uso:
    python -m machado_gpt.validacao
"""

import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
N = 8


def palavras(texto):
    return re.findall(r"\w+", texto.lower())


def ngramas(ws, n=N):
    return {hash(tuple(ws[i:i + n])) for i in range(len(ws) - n + 1)}


def main():
    corpus = (RAIZ / "corpus" / "machado_obra_completa.txt").read_text(encoding="utf-8")
    nosso = ngramas(palavras(corpus))

    csv.field_size_limit(sys.maxsize)
    with open(RAIZ / "archive" / "obras_machado_de_assis.csv", encoding="utf-8") as f:
        linhas = list(csv.DictReader(f))

    total, achados = defaultdict(int), defaultdict(int)
    for linha in linhas:
        g = ngramas(palavras(linha["texto"]))
        total[linha["categoria"]] += len(g)
        achados[linha["categoria"]] += len(g & nosso)

    resultado = {cat: round(achados[cat] / total[cat], 4) for cat in sorted(total)}
    sem_trad = [c for c in total if c != "tradução"]
    resultado["total_sem_traducoes"] = round(
        sum(achados[c] for c in sem_trad) / sum(total[c] for c in sem_trad), 4)
    print(json.dumps(resultado, ensure_ascii=False, indent=2))
    saida = RAIZ / "results" / "validacao_corpus.json"
    saida.parent.mkdir(exist_ok=True)
    saida.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
