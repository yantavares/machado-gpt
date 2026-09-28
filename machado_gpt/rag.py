"""Protótipo de extensão: responder perguntas sobre o universo de Machado.

Autoria: código original do aluno.

Um modelo de linguagem treinado só para continuar texto não responde
perguntas: ele imita a prosa, sem ligar a pergunta aos fatos das obras. A
extensão proposta segue a geração aumentada por recuperação (RAG, Lewis et al.,
2020) em duas etapas:

1. Recuperação: as obras são cortadas em passagens de ~600 caracteres,
   indexadas por TF-IDF (unigramas e bigramas de palavras). A pergunta é
   comparada às passagens por similaridade de cosseno.
2. Geração: as passagens recuperadas entram no prompt do modelo ajustado
   (GPorTuguese-2 + Machado), seguidas de "Pergunta: ... Resposta:".

Um conjunto de 20 perguntas com a obra de origem e a palavra-chave da resposta
mede a recuperação (acerto@k: a palavra-chave aparece em alguma das k
passagens; obra@k: alguma passagem vem da obra certa). A geração é avaliada
qualitativamente, e o artigo discute por que ela exige ajuste por instruções.

Uso:
    python -m machado_gpt.rag --avaliar
    python -m machado_gpt.rag --pergunta "Quem é Capitu?" [--run runs/ptbr_ft]
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
TAM_PASSAGEM = 600

# pergunta, arquivo da obra, palavra-chave que deve aparecer na passagem
PERGUNTAS = [
    ("Quem era o melhor amigo de Bentinho no seminário?", "romance/marm08.txt", "Escobar"),
    ("Como se chamava o filho de Capitu e Bentinho?", "romance/marm08.txt", "Ezequiel"),
    ("Onde Bento Santiago mandou construir a casa em que mora?", "romance/marm08.txt", "Engenho Novo"),
    ("A quem Brás Cubas dedica as suas memórias póstumas?", "romance/marm05.txt", "verme"),
    ("Que remédio Brás Cubas queria inventar para aliviar a melancolia da humanidade?", "romance/marm05.txt", "emplasto"),
    ("Qual é a filosofia criada por Quincas Borba?", "romance/marm05.txt", "Humanitismo"),
    ("Por quem Rubião se apaixona no Rio de Janeiro?", "romance/marm07.txt", "Sofia"),
    ("Qual o nome do médico que funda o hospício em O Alienista?", "contos/macn003.txt", "Bacamarte"),
    ("Qual era o posto militar de Jacobina no conto O Espelho?", "contos/macn003.txt", "alferes"),
    ("Quem sente prazer com o sofrimento alheio em A Causa Secreta?", "contos/macn005.txt", "Fortunato"),
    ("Quem vai consultar a cartomante com medo de ser descoberto?", "contos/macn005.txt", "Camilo"),
    ("Com quem Nogueira conversa na noite de Natal em Missa do Galo?", "contos/macn006.txt", "Conceição"),
    ("Qual é o ofício de Cândido Neves em Pai contra mãe?", "contos/macn007.txt", "escravos fugidos"),
    ("Como se chamam os gêmeos rivais filhos de Natividade?", "romance/marm09.txt", "Pedro"),
    ("Quem é a madrasta de Iaiá Garcia?", "romance/marm04.txt", "Estela"),
    ("Quem é o diplomata aposentado que escreve o diário de 1888 e 1889?", "romance/marm10.txt", "Aires"),
    ("Como se chama a filha natural que o Conselheiro Vale reconhece em testamento?", "romance/marm03.txt", "Helena"),
    ("Em que vila Simão Bacamarte abre a Casa Verde?", "contos/macn003.txt", "Itaguaí"),
    ("Que ofício o pai recomenda a Janjão quando ele completa vinte e um anos?", "contos/macn003.txt", "medalhão"),
    ("Como José Dias descreve os olhos de Capitu?", "romance/marm08.txt", "oblíqua"),
]

STOP = set("""a o e é de do da dos das em no na nos nas um uma uns umas que quem qual quais
como onde por para com se ao aos à às seu sua seus suas ele ela eles elas lhe não mais
foi era ser são o os as mas ou já muito também quando""".split())


def passagens():
    """Corta cada obra em passagens de ~TAM_PASSAGEM caracteres."""
    saida = []
    with open(RAIZ / "corpus" / "obras.jsonl", encoding="utf-8") as f:
        for linha in f:
            o = json.loads(linha)
            atual = []
            for p in o["paragrafos"]:
                atual.append(p)
                if sum(len(x) for x in atual) >= TAM_PASSAGEM:
                    saida.append({"arquivo": o["arquivo"], "titulo": o["titulo"], "ano": o["ano"],
                                  "texto": "\n".join(atual)})
                    atual = []
            if atual:
                saida.append({"arquivo": o["arquivo"], "titulo": o["titulo"], "ano": o["ano"],
                              "texto": "\n".join(atual)})
    return saida


class Recuperador:
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.passagens = passagens()
        self.vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, max_df=0.5,
                                   sublinear_tf=True, stop_words=list(STOP),
                                   token_pattern=r"(?u)\b\w\w+\b")
        self.matriz = self.vec.fit_transform(p["texto"] for p in self.passagens)

    def busca(self, pergunta, k=5):
        q = self.vec.transform([pergunta])
        sims = (self.matriz @ q.T).toarray().ravel()
        top = np.argsort(-sims)[:k]
        return [{**self.passagens[i], "score": float(sims[i])} for i in top]


def avalia_recuperacao(rec, ks=(1, 5, 10)):
    res = {f"acerto@{k}": 0 for k in ks} | {f"obra@{k}": 0 for k in ks}
    detalhes = []
    for pergunta, obra, chave in PERGUNTAS:
        top = rec.busca(pergunta, max(ks))
        pos_chave = next((i for i, p in enumerate(top) if chave.lower() in p["texto"].lower()), None)
        pos_obra = next((i for i, p in enumerate(top) if p["arquivo"] == obra), None)
        for k in ks:
            res[f"acerto@{k}"] += pos_chave is not None and pos_chave < k
            res[f"obra@{k}"] += pos_obra is not None and pos_obra < k
        detalhes.append({"pergunta": pergunta, "obra": obra, "chave": chave,
                         "rank_chave": pos_chave, "rank_obra": pos_obra,
                         "top1": {"titulo": top[0]["titulo"], "texto": top[0]["texto"][:300]}})
    n = len(PERGUNTAS)
    return {k: v / n for k, v in res.items()} | {"n_perguntas": n, "detalhes": detalhes}


def monta_prompt(pergunta, top, max_chars=1400):
    contexto, total = [], 0
    for p in top:
        trecho = p["texto"][:max_chars - total]
        contexto.append(f"[{p['titulo']}, {p['ano']}]\n{trecho}")
        total += len(trecho)
        if total >= max_chars:
            break
    return "\n\n".join(contexto) + f"\n\nPergunta: {pergunta}\nResposta:"


def chave_no_prompt(rec, k=3):
    """Em quais perguntas a palavra da resposta chegou ao prompt do gerador."""
    return {p: c.lower() in monta_prompt(p, rec.busca(p, k)).lower() for p, _, c in PERGUNTAS}


def responde(modelo, tok, pergunta, rec, device="cpu", k=3, max_tokens=60):
    import torch
    top = rec.busca(pergunta, k)
    prompt = monta_prompt(pergunta, top)
    ids = tok.encode(prompt)[-(modelo.block_size - max_tokens):]
    x = torch.tensor([ids], dtype=torch.long, device=device)
    torch.manual_seed(1337)
    y = modelo.gera(x, max_tokens, temperature=0.3, top_k=20)
    resposta = tok.decode(y[0, len(ids):].tolist())
    return {"pergunta": pergunta, "fontes": [f"{p['titulo']} ({p['ano']})" for p in top],
            # primeira linha não vazia: o modelo às vezes começa com uma quebra de linha
            "resposta": next((l.strip() for l in resposta.split("\n") if l.strip()), "")}


def main():
    ap = argparse.ArgumentParser(description="Recuperação + geração sobre a obra de Machado.")
    ap.add_argument("--avaliar", action="store_true", help="mede a recuperação nas 20 perguntas")
    ap.add_argument("--pergunta")
    ap.add_argument("--run", help="modelo gerador (ex.: runs/ptbr_ft); omita para só recuperar")
    ap.add_argument("--gerar-todas", action="store_true", help="gera respostas para as 20 perguntas")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    rec = Recuperador()
    saida = RAIZ / "results" / "rag.json"
    resultado = json.loads(saida.read_text()) if saida.exists() else {}
    if args.avaliar:
        r = avalia_recuperacao(rec)
        resultado["recuperacao"] = r
        print(json.dumps({k: v for k, v in r.items() if k != "detalhes"}, indent=2))

    if args.pergunta or args.gerar_todas:
        modelo = tok = None
        if args.run:
            from machado_gpt import tokenizacao
            from machado_gpt.avaliacao import NANOGPT, carrega_run
            modelo = carrega_run(args.run, args.device)
            tok = tokenizacao.carrega(NANOGPT / "data" / modelo.dataset)
        perguntas = [args.pergunta] if args.pergunta else [p for p, _, _ in PERGUNTAS]
        respostas = []
        for p in perguntas:
            if modelo is None:
                top = rec.busca(p, 3)
                r = {"pergunta": p, "fontes": [f"{t['titulo']} ({t['ano']})" for t in top],
                     "passagem": top[0]["texto"][:400]}
            else:
                r = responde(modelo, tok, p, rec, args.device)
            respostas.append(r)
            print(json.dumps(r, ensure_ascii=False, indent=2))
        if args.gerar_todas and modelo is not None:
            chaves = {p: c for p, _, c in PERGUNTAS}
            no_prompt = chave_no_prompt(rec)
            for r in respostas:
                r["contem_chave"] = chaves[r["pergunta"]].lower() in r["resposta"].lower()
                r["chave_no_prompt"] = no_prompt[r["pergunta"]]
            acertos = sum(r["contem_chave"] for r in respostas)
            print(f"respostas com a palavra-chave: {acertos}/{len(respostas)}; "
                  f"palavra-chave presente no prompt: {sum(no_prompt.values())}/{len(respostas)}")
            resultado["geracao"] = {"modelo": args.run, "acertos": acertos,
                                    "chave_no_prompt": sum(no_prompt.values()), "respostas": respostas}

    saida.parent.mkdir(exist_ok=True)
    saida.write_text(json.dumps(resultado, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
