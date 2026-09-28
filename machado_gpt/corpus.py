"""Constrói o corpus de treino a partir da Obra Completa de Machado de Assis.

Autoria: código original do aluno.

Fonte: os 246 arquivos de machado.mec.gov.br em `machado/` (o mesmo acervo do
corpus `machado` do NLTK). Etapas:

1. Decodifica em Windows-1252. O acervo não é Latin-1 puro: o travessão dos
   diálogos é o byte 0x97, que em cp1252 é "—" e em Latin-1 vira um caractere
   de controle invisível. O mesmo vale para aspas curvas (0x93/0x94) e "…".
2. Separa parágrafos (linhas em branco) e desfaz a quebra de linha rígida
   dentro de cada parágrafo, para que o modelo aprenda quebras de parágrafo e
   não a largura de coluna do arquivo.
   Parte dos arquivos grafa o travessão como "--"; ele é unificado em "—".
3. Remove o cabeçalho editorial (gênero, "Texto-fonte", "Publicado
   originalmente...") e os índices (sumários), mantendo o título da obra.
4. Remove parágrafos longos repetidos entre arquivos, para evitar vazamento
   entre treino e teste.
5. Divide em treino/validação/teste por blocos de ~4.000 caracteres,
   intercalados ao longo de todas as obras (18/1/1 a cada 20 blocos). Assim os
   três conjuntos cobrem todos os gêneros e épocas, e nenhum trecho contínuo
   aparece em dois conjuntos.

As traduções (Oliver Twist, Os Trabalhadores do Mar, Suplício de uma Mulher)
ficam de fora por padrão, porque não são prosa original de Machado.

Uso:
    python -m machado_gpt.corpus                 # gera corpus/
    python -m machado_gpt.corpus --incluir-traducoes
"""

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path

RAIZ = Path(__file__).resolve().parent.parent
ACERVO = RAIZ / "machado"
SAIDA = RAIZ / "corpus"

# pasta do acervo -> nome do gênero usado no manifesto
GENEROS = {
    "romance": "romance",
    "contos": "conto",
    "cronica": "crônica",
    "critica": "crítica",
    "poesia": "poesia",
    "teatro": "teatro",
    "miscelanea": "miscelânea",
    "traducao": "tradução",
}

# Seis arquivos não têm o parágrafo "Publicado originalmente...", que marca o
# fim do cabeçalho; para eles o número de parágrafos de cabeçalho foi
# verificado à mão.
CABECALHO_MANUAL = {
    "critica/mact23.txt": 6,
    "critica/mact28.txt": 6,
    "critica/mact31.txt": 6,
    "critica/mact37.txt": 6,
    "critica/mact45.txt": 6,
    "poesia/maps07.txt": 7,
}

TAM_BLOCO = 4000        # caracteres por bloco na divisão treino/val/teste
CICLO = 20              # a cada 20 blocos: 18 treino, 1 validação, 1 teste
MIN_DUPLICATA = 100     # só parágrafos com pelo menos isso contam como duplicata


def normaliza(texto):
    """Normaliza espaços e a forma Unicode de um parágrafo."""
    texto = unicodedata.normalize("NFC", texto)
    texto = texto.replace("­", "")          # hífen suave
    texto = texto.replace(" ", " ").replace("\t", " ")
    # parte dos arquivos grafa o travessão como "--"; unifica em "—"
    texto = re.sub(r"-{2,}", "—", texto)
    return re.sub(r"\s+", " ", texto).strip()


def paragrafos(texto):
    """Parágrafos separados por linha em branco, já sem quebras internas."""
    texto = texto.replace("\r", "")
    return [normaliza(p) for p in re.split(r"\n[ \t ]*\n", texto) if p.strip()]


def chave_indice(p):
    """Forma canônica de uma entrada de índice, para achar sua repetição."""
    return re.sub(r"[\s.:;,!?-]+$", "", p.upper()).strip()


def eh_indice(p):
    return chave_indice(p) in {"ÍNDICE", "INDICE"}


def remove_indices(pars, log):
    """Remove sumários ("ÍNDICE" seguido de entradas curtas).

    O texto recomeça quando uma entrada do índice se repete (o título do
    primeiro conto ou o primeiro capítulo). Se não houver repetição antes de
    um parágrafo longo, descarta só o bloco de entradas curtas.
    """
    saida, i = [], 0
    while i < len(pars):
        if not eh_indice(pars[i]):
            saida.append(pars[i])
            i += 1
            continue
        j, entradas, achou = i + 1, [], False
        while j < len(pars) and len(pars[j]) <= 150:
            chave = chave_indice(pars[j])
            if entradas:
                primeira = chave_indice(entradas[0])
                # repetição exata, título seguido de subtítulo, ou o inverso
                # ("CAPÍTULO PRIMEIRO" no texto x "CAPÍTULO PRIMEIRO - TRÊS AMIGOS" no índice)
                if (chave == primeira or chave.startswith(primeira + " ")
                        or (len(chave) >= 8 and primeira.startswith(chave + " "))):
                    achou = True
                    break
            if eh_indice(pars[j]):
                break
            entradas.append(pars[j])
            j += 1
        log.append({"entradas": len(entradas), "repeticao": achou,
                    "primeira": entradas[0] if entradas else ""})
        i = j
    return saida


def le_obra(caminho):
    rel = caminho.relative_to(ACERVO).as_posix()
    bruto = caminho.read_bytes().decode("cp1252")
    pars = paragrafos(bruto)

    # 1a linha: "Gênero, Título, Ano" (o ano às vezes vem colado ou com ".htm")
    anos = re.findall(r"1[89]\d\d", pars[0])
    ano = int(anos[0]) if anos else None
    titulo = pars[1]

    if rel in CABECALHO_MANUAL:
        inicio = CABECALHO_MANUAL[rel]
    else:
        pub = [i for i, p in enumerate(pars[:13]) if re.match(r"Publicad[oa]s?\b", p)]
        if not pub:
            raise ValueError(f"{rel}: cabeçalho não reconhecido")
        inicio = pub[0] + 1

    log = []
    corpo = remove_indices(pars[inicio:], log)
    return {
        "arquivo": rel,
        "genero": GENEROS[rel.split("/")[0]],
        "titulo": titulo,
        "ano": ano,
        "paragrafos": [titulo] + corpo,
        "indices": log,
    }


def hash_par(p):
    return hashlib.sha1(p.encode("utf-8")).hexdigest()


def constroi(incluir_traducoes=False, saida=SAIDA):
    arquivos = sorted(ACERVO.glob("*/*.txt"))
    obras = [le_obra(a) for a in arquivos]
    if not incluir_traducoes:
        obras = [o for o in obras if o["genero"] != "tradução"]

    # remove parágrafos longos que já apareceram em outra obra
    vistos, duplicatas = {}, []
    for o in obras:
        mantidos = []
        for p in o["paragrafos"]:
            if len(p) >= MIN_DUPLICATA:
                h = hash_par(p)
                if h in vistos:
                    duplicatas.append({"arquivo": o["arquivo"], "original": vistos[h],
                                       "inicio": p[:80]})
                    continue
                vistos[h] = o["arquivo"]
            mantidos.append(p)
        o["paragrafos"] = mantidos

    # divisão em blocos intercalados
    splits = {"train": [], "val": [], "test": []}
    g = 0
    for o in obras:
        o["chars"] = {"train": 0, "val": 0, "test": 0}
        bloco = []
        tamanho = 0
        blocos = []
        for p in o["paragrafos"]:
            bloco.append(p)
            tamanho += len(p) + 1
            if tamanho >= TAM_BLOCO:
                blocos.append(bloco)
                bloco, tamanho = [], 0
        if bloco:
            blocos.append(bloco)
        for b in blocos:
            r = g % CICLO
            nome = "val" if r == CICLO - 2 else "test" if r == CICLO - 1 else "train"
            texto = "\n".join(b)
            splits[nome].append((o["arquivo"], texto))
            o["chars"][nome] += len(texto) + 1
            g += 1

    saida.mkdir(parents=True, exist_ok=True)
    completo = "\n\n".join("\n".join(o["paragrafos"]) for o in obras) + "\n"
    (saida / "machado_obra_completa.txt").write_text(completo, encoding="utf-8")

    for nome, blocos in splits.items():
        # blocos consecutivos da mesma obra são unidos por "\n"; troca de obra, "\n\n"
        partes, anterior = [], None
        for arq, texto in blocos:
            if anterior is not None:
                partes.append("\n\n" if arq != anterior else "\n")
            partes.append(texto)
            anterior = arq
        (saida / f"{nome}.txt").write_text("".join(partes) + "\n", encoding="utf-8")

    # obras com metadados, usadas pela recuperação de passagens (rag.py)
    with open(saida / "obras.jsonl", "w", encoding="utf-8") as f:
        for o in obras:
            f.write(json.dumps({k: o[k] for k in ("arquivo", "genero", "titulo", "ano", "paragrafos")},
                               ensure_ascii=False) + "\n")

    # teste separado por gênero, para medir onde cada modelo erra mais
    genero_de = {o["arquivo"]: o["genero"] for o in obras}
    por_gen = {}
    for arq, texto in splits["test"]:
        por_gen.setdefault(genero_de[arq], []).append(texto)
    (saida / "test_por_genero").mkdir(exist_ok=True)
    for gen, textos_gen in por_gen.items():
        (saida / "test_por_genero" / f"{gen}.txt").write_text(
            "\n\n".join(textos_gen) + "\n", encoding="utf-8")

    with open(saida / "manifesto.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["arquivo", "genero", "titulo", "ano", "paragrafos", "caracteres",
                    "chars_treino", "chars_val", "chars_teste"])
        for o in obras:
            w.writerow([o["arquivo"], o["genero"], o["titulo"], o["ano"], len(o["paragrafos"]),
                        sum(len(p) + 1 for p in o["paragrafos"]),
                        o["chars"]["train"], o["chars"]["val"], o["chars"]["test"]])

    textos = {n: (saida / f"{n}.txt").read_text(encoding="utf-8") for n in splits}
    freq = Counter(completo)
    por_genero = Counter()
    for o in obras:
        por_genero[o["genero"]] += sum(len(p) + 1 for p in o["paragrafos"])
    stats = {
        "obras": len(obras),
        "incluir_traducoes": incluir_traducoes,
        "caracteres_total": len(completo),
        "palavras_total": len(re.findall(r"\w+", completo)),
        "caracteres_por_split": {n: len(t) for n, t in textos.items()},
        "caracteres_por_genero": dict(por_genero.most_common()),
        "vocabulario_chars": len(freq),
        "chars_raros": {c: n for c, n in freq.items() if n < 5},
        "duplicatas_removidas": len(duplicatas),
        "indices_removidos": sum(len(o["indices"]) for o in obras),
        "indices_sem_repeticao": [
            {"arquivo": o["arquivo"], **l} for o in obras for l in o["indices"] if not l["repeticao"]
        ],
    }
    (saida / "estatisticas.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                                             encoding="utf-8")
    (saida / "duplicatas.json").write_text(json.dumps(duplicatas, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--incluir-traducoes", action="store_true")
    ap.add_argument("--saida", type=Path, default=SAIDA)
    args = ap.parse_args()
    s = constroi(args.incluir_traducoes, args.saida)
    resumo = {k: v for k, v in s.items() if k not in ("chars_raros", "indices_sem_repeticao")}
    print(json.dumps(resumo, ensure_ascii=False, indent=2))
    print(f"índices sem repetição (revisar): {len(s['indices_sem_repeticao'])}")


if __name__ == "__main__":
    main()
