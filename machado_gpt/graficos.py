"""Figuras e tabela de resultados para o artigo.

Autoria: código original do aluno.

Lê `results/metrics/*.json` e `results/amostras/*.json` e gera, em
`paper/figuras/`:
- curvas.pdf: bpc de validação ao longo do treino (modelos char e BPE);
- bpc_teste.pdf: bpc no teste de todos os modelos e linhas de base;
- generos.pdf: bpc por gênero literário dos melhores modelos;
e `results/tabela_resultados.csv` / `.md` com todas as métricas.

Cores: paleta categórica de referência (slots 1-3, validados para daltonismo
em todos os pares), sempre com marcador ou estilo de linha como segunda
codificação, para leitura em impressão em tons de cinza.

Uso:
    python -m machado_gpt.graficos
"""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402

# eixos com vírgula decimal (o artigo é em português)
VIRGULA = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))

RAIZ = Path(__file__).resolve().parent.parent
RES = RAIZ / "results"
FIG = RAIZ / "paper" / "figuras"
NANOGPT = RAIZ / "nanoGPT-master"

AZUL, LARANJA, VERDE_AGUA = "#2a78d6", "#eb6834", "#1baf7a"
CINZA, TEXTO, TEXTO2, GRADE = "#8a8984", "#0b0b0b", "#52514e", "#e6e5e1"

NOMES = {
    "unigrama": "Unigrama",
    "bigrama": "Bigrama (vídeo)",
    "trigrama": "Trigrama",
    "char_cpu": "Char 0,8M (CPU)",
    "video_gpt": "GPT do vídeo 10,8M",
    "char_baby": "nanoGPT char 10,8M",
    "char_large": "nanoGPT char 25M",
    "gpt2_scratch": "BPE GPT-2, do zero",
    "ptbr_scratch": "BPE pt, do zero",
    "gpt2_zeroshot": "GPT-2 124M zero-shot",
    "ptbr_zeroshot": "GPorTuguese-2 zero-shot",
    "gpt2_ft": "GPT-2 124M ajustado",
    "ptbr_ft": "GPorTuguese-2 ajustado",
}
ORDEM = ["unigrama", "bigrama", "trigrama", "char_cpu", "video_gpt", "char_baby", "char_large",
         "gpt2_scratch", "ptbr_scratch", "gpt2_zeroshot", "ptbr_zeroshot", "gpt2_ft", "ptbr_ft"]
GENEROS = ["romance", "conto", "crônica", "crítica", "teatro", "poesia"]  # miscelânea: amostra pequena


def estilo():
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": CINZA, "axes.labelcolor": TEXTO2, "xtick.color": TEXTO2,
        "ytick.color": TEXTO2, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRADE, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "lines.linewidth": 1.4, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def salva(fig, nome):
    """PDF vetorial para o artigo e PNG para o notebook."""
    fig.savefig(FIG / f"{nome}.pdf")
    fig.savefig(FIG / f"{nome}.png", dpi=200)


def carrega():
    met = {}
    for arq in sorted((RES / "metrics").glob("*.json")):
        if arq.stem == "baselines_ngrama":
            for nome, r in json.loads(arq.read_text()).items():
                met[nome] = r
        else:
            met[arq.stem] = json.loads(arq.read_text())
    amostras = {}
    for arq in sorted((RES / "amostras").glob("*.json")):
        amostras[arq.stem] = json.loads(arq.read_text())["metricas"]
    return met, amostras


def chars_por_token(dataset):
    info = json.loads((NANOGPT / "data" / dataset / "info.json").read_text())
    return info["val"]["caracteres_por_token"]


def curva_bpc(r):
    """Perda de validação do train.py (nats/token) convertida para bpc aproximado."""
    cpt = chars_por_token(r["dataset"])
    ev = r["treino"].get("evals") or r["treino"].get("log")
    return [e["iter"] for e in ev], [e["val"] / cpt / 0.6931471805599453 for e in ev]


def fig_curvas(met):
    fig, eixos = plt.subplots(1, 2, figsize=(7.0, 1.85), sharey=False)
    paineis = [
        ("(a) Caracteres, do zero", [("video_gpt", AZUL, "o", "-"), ("char_baby", LARANJA, "s", "--"),
                                      ("char_large", VERDE_AGUA, "^", ":")]),
        ("(b) BPE, contexto de 256 a 512 tokens", [("gpt2_scratch", AZUL, "o", "-"),
                                                   ("ptbr_scratch", LARANJA, "s", "--"),
                                                   ("ptbr_ft", VERDE_AGUA, "^", ":"),
                                                   ("gpt2_ft", CINZA, "D", "-.")]),
    ]
    for ax, (titulo, series) in zip(eixos, paineis):
        for nome, cor, marc, ls in series:
            if nome not in met or "treino" not in met[nome]:
                continue
            x, y = curva_bpc(met[nome])
            x, y = x[1:], y[1:]  # a avaliação no passo 0 (modelo aleatório) achata o gráfico
            ax.plot(x, y, color=cor, marker=marc, markersize=3, linestyle=ls, label=NOMES[nome])
        ax.set_title(titulo, loc="left", color=TEXTO)
        ax.set_xlabel("iteração")
        ax.set_ylabel("bpc de validação (aprox.)")
        ax.yaxis.set_major_formatter(VIRGULA)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(frameon=False)
    salva(fig, "curvas")
    plt.close(fig)


def fig_bpc(met):
    nomes = [n for n in ORDEM if n in met]
    vals = [met[n]["test"]["bpc"] for n in nomes]
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    ys = range(len(nomes))[::-1]
    melhor = min(vals)
    for y, n, v in zip(ys, nomes, vals):
        cor = CINZA if n in ("unigrama", "bigrama", "trigrama") else AZUL
        ax.barh(y, v, color=LARANJA if v == melhor else cor, height=0.62)
        ax.text(v + 0.04, y, f"{v:.2f}".replace(".", ","), va="center", color=TEXTO, fontsize=6.5)
    ax.set_yticks(list(ys), [NOMES[n] for n in nomes])
    ax.set_xlabel("bits por caractere no teste (menor é melhor)")
    ax.set_xlim(0, max(vals) * 1.12)
    ax.grid(axis="y", visible=False)
    salva(fig, "bpc_teste")
    plt.close(fig)


def fig_generos(met):
    series = [("char_baby", AZUL, "o"), ("ptbr_scratch", LARANJA, "s"), ("ptbr_ft", VERDE_AGUA, "^")]
    series = [s for s in series if s[0] in met]
    fig, ax = plt.subplots(figsize=(3.5, 2.1))
    xs = range(len(GENEROS))
    for k, (nome, cor, marc) in enumerate(series):
        ys = [met[nome]["test_por_genero"][g]["bpc"] for g in GENEROS]
        desloc = (k - (len(series) - 1) / 2) * 0.18
        ax.plot([x + desloc for x in xs], ys, linestyle="none", marker=marc, markersize=5,
                color=cor, label=NOMES[nome])
    ax.set_xticks(list(xs), GENEROS)
    ax.set_ylabel("bpc no teste")
    ax.yaxis.set_major_formatter(VIRGULA)
    ax.legend(frameon=False, loc="upper left", ncol=1)
    ax.grid(axis="x", visible=False)
    salva(fig, "generos")
    plt.close(fig)


def tabela(met, amostras):
    ref = amostras.get("referencia_humana", {})
    linhas = []
    for n in ORDEM:
        if n not in met:
            continue
        r, a = met[n], amostras.get(n, {})
        tr = r.get("treino", {})
        linhas.append({
            "modelo": NOMES[n], "run": n,
            "parametros_M": round(r.get("parametros", 0) / 1e6, 2),
            "contexto": r.get("block_size", ""),
            "bpc_val": round(r["val"]["bpc"], 4), "bpc_teste": round(r["test"]["bpc"], 4),
            "ppl_palavra": round(r["test"]["ppl_palavra"], 1) if "ppl_palavra" in r["test"] else "",
            "min_treino": round(tr.get("segundos_total", 0) / 60, 1) if tr else "",
            "ms_por_iter": tr.get("ms_por_iter_mediana") or tr.get("ms_por_iter", ""),
            **{k: round(v, 4) for k, v in a.items() if k != "palavras"},
        })
    if ref:
        linhas.append({"modelo": "Texto real (teste)", "run": "referencia",
                       **{k: round(v, 4) for k, v in ref.items() if k != "palavras"}})
    campos = list(dict.fromkeys(k for l in linhas for k in l))
    with open(RES / "tabela_resultados.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(linhas)
    md = ["| " + " | ".join(campos) + " |", "|" + "---|" * len(campos)]
    md += ["| " + " | ".join(str(l.get(c, "")) for c in campos) + " |" for l in linhas]
    (RES / "tabela_resultados.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return linhas


TOKENIZADOR = {"machado_char": "char", "machado_gpt2": "BPE en", "machado_ptbr": "BPE pt"}
INICIO = {"gpt2_zeroshot": "GPT-2", "ptbr_zeroshot": "GPorT.-2", "gpt2_ft": "GPT-2",
          "ptbr_ft": "GPorT.-2"}


def br(x, casas):
    """Número no formato brasileiro (vírgula decimal, ponto de milhar)."""
    s = f"{x:,.{casas}f}"
    return s.replace(",", "_").replace(".", ",").replace("_", ".")


def tabelas_latex(met, amostras):
    """Linhas das tabelas do artigo, lidas de results/ (nada digitado à mão)."""
    TAB = RAIZ / "paper" / "tabelas"
    TAB.mkdir(parents=True, exist_ok=True)
    melhor = min(met[n]["test"]["bpc"] for n in ORDEM if n in met)
    linhas = []
    for n in ORDEM:
        if n not in met:
            continue
        r = met[n]
        bpc = br(r["test"]["bpc"], 3)
        if r["test"]["bpc"] == melhor:
            bpc = r"\textbf{" + bpc + "}"
        if n in ("unigrama", "bigrama", "trigrama"):
            # n-gramas de contagem: sem treino por gradiente nem perplexidade por palavra
            colunas = [NOMES[n], "char", "--", "--", "contagem", "--", bpc, "--"]
        else:
            tr = r.get("treino", {})
            colunas = [NOMES[n], TOKENIZADOR[r["dataset"]], br(r["parametros"] / 1e6, 1),
                       str(r["block_size"]), INICIO.get(n, "aleatório"),
                       br(tr["segundos_total"] / 60, 0) if tr else "0", bpc,
                       br(r["test"]["ppl_palavra"], 0)]
        linhas.append(" & ".join(colunas) + r" \\")
    # o \bottomrule vai no próprio arquivo: após um \input, ele dá "Misplaced \noalign"
    (TAB / "resultados.tex").write_text("\n".join(linhas) + "\n\\bottomrule\n", encoding="utf-8")

    sel = [n for n in ["char_cpu", "video_gpt", "char_baby", "char_large", "gpt2_scratch", "ptbr_scratch",
                       "gpt2_zeroshot", "ptbr_zeroshot", "gpt2_ft", "ptbr_ft", "referencia_humana"]
           if n in amostras]
    tx = []
    for n in sel:
        a = amostras[n]
        nome = "Texto real (teste)" if n == "referencia_humana" else NOMES[n]
        tx.append(f"{nome} & {br(100 * a['palavras_validas'], 1)} & {br(a['distinct_2'], 3)} & "
                  f"{br(100 * a['memorizacao_8grama'], 1)} & {br(100 * a['repeticao_4grama'], 1)} \\\\")
        if n == "ptbr_ft":
            tx.append(r"\midrule")
    (TAB / "texto.tex").write_text("\n".join(tx) + "\n\\bottomrule\n", encoding="utf-8")


def tabela_slides(met):
    """Tabela resumida da apresentação (apresentacao/), lida de results/."""
    TAB = RAIZ / "paper" / "tabelas"
    sel = ["bigrama", "char_cpu", "video_gpt", "char_baby", "char_large", "ptbr_scratch",
           "gpt2_zeroshot", "ptbr_zeroshot", "gpt2_ft", "ptbr_ft"]
    melhor = min(met[n]["test"]["bpc"] for n in sel if n in met)
    linhas = []
    for n in sel:
        if n not in met:
            continue
        r = met[n]
        tr = r.get("treino")
        params = br(r["parametros"] / 1e6, 1) if "parametros" in r and n != "bigrama" else "--"
        tempo = br(tr["segundos_total"] / 60, 0) if tr else "--"
        bpc = br(r["test"]["bpc"], 3)
        if r["test"]["bpc"] == melhor:
            bpc = r"\textbf{" + bpc + "}"
        linhas.append(f"{NOMES[n]} & {params} & {tempo} & {bpc} \\\\")
    (TAB / "slides.tex").write_text("\n".join(linhas) + "\n\\bottomrule\n", encoding="utf-8")


def main():
    estilo()
    FIG.mkdir(parents=True, exist_ok=True)
    met, amostras = carrega()
    fig_curvas(met)
    fig_bpc(met)
    if all(n in met for n in ("char_baby",)):
        fig_generos(met)
    tabelas_latex(met, amostras)
    tabela_slides(met)
    for l in tabela(met, amostras):
        print(l)


if __name__ == "__main__":
    main()
