"""Tokenizadores usados nos experimentos e preparação dos .bin do nanoGPT.

Autoria: código original do aluno. A lógica de exportação (ids em uint16,
`train.bin`/`val.bin`, `meta.pkl` com `stoi`/`itos`) segue o formato de
`nanoGPT-master/data/shakespeare_char/prepare.py` e
`data/shakespeare/prepare.py` (Karpathy, MIT), para que o `train.py` original
leia os dados sem alteração.

Três tokenizações do mesmo texto:
- `char`: um token por caractere (vocabulário de ~144 símbolos), como no vídeo
  "Let's build GPT" e em `shakespeare_char`;
- `gpt2`: BPE do GPT-2 (tiktoken), treinado em texto em inglês;
- `ptbr`: BPE do GPorTuguese-2 (pierreguillou/gpt2-small-portuguese), o GPT-2
  small ajustado na Wikipédia em português, que tem vocabulário próprio.

Cada diretório de dados recebe também `test.bin` (ignorado pelo train.py) e
`tokenizador.json`, que diz aos scripts de avaliação como decodificar.

Uso:
    python -m machado_gpt.tokenizacao char gpt2 ptbr
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
CORPUS = RAIZ / "corpus"
NANOGPT = RAIZ / "nanoGPT-master"
HF_PTBR = "pierreguillou/gpt2-small-portuguese"

DATASETS = {"char": "machado_char", "gpt2": "machado_gpt2", "ptbr": "machado_ptbr"}


class TokChar:
    def __init__(self, stoi):
        self.stoi = stoi
        self.itos = {i: c for c, i in stoi.items()}
        self.vocab_size = len(stoi)

    @classmethod
    def do_texto(cls, texto):
        return cls({c: i for i, c in enumerate(sorted(set(texto)))})

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, ids):
        return "".join(self.itos[i] for i in ids)


class TokTiktoken:
    def __init__(self, nome="gpt2"):
        import tiktoken
        self.enc = tiktoken.get_encoding(nome)
        self.vocab_size = self.enc.n_vocab
        self.eot = self.enc.eot_token

    def encode(self, s):
        return self.enc.encode_ordinary(s)

    def decode(self, ids):
        return self.enc.decode(ids)


class TokHF:
    def __init__(self, nome=HF_PTBR):
        from transformers import AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(nome)
        self.vocab_size = len(self.tok)

    def encode(self, s):
        # codifica linha a linha (com o "\n" final): o pré-tokenizador do GPT-2
        # já separa "\n" das palavras, então o resultado é o mesmo do texto
        # inteiro, mas sem o aviso de sequência longa do transformers
        linhas = s.splitlines(keepends=True)
        ids = []
        for i in range(0, len(linhas), 20000):
            for seq in self.tok(linhas[i:i + 20000], add_special_tokens=False)["input_ids"]:
                ids.extend(seq)
        return ids

    def decode(self, ids):
        return self.tok.decode(ids, clean_up_tokenization_spaces=False)


def carrega(data_dir):
    """Reconstrói o tokenizador de um diretório de dados."""
    data_dir = Path(data_dir)
    info = json.loads((data_dir / "tokenizador.json").read_text())
    if info["tipo"] == "char":
        with open(data_dir / "meta.pkl", "rb") as f:
            return TokChar(pickle.load(f)["stoi"])
    if info["tipo"] == "tiktoken":
        return TokTiktoken(info["nome"])
    return TokHF(info["nome"])


def prepara(tipo, corpus=CORPUS, destino=None):
    destino = Path(destino or NANOGPT / "data" / DATASETS[tipo])
    destino.mkdir(parents=True, exist_ok=True)
    textos = {n: (corpus / f"{n}.txt").read_text(encoding="utf-8") for n in ("train", "val", "test")}

    if tipo == "char":
        tok = TokChar.do_texto("".join(textos.values()))
        info_tok = {"tipo": "char"}
        with open(destino / "meta.pkl", "wb") as f:
            pickle.dump({"vocab_size": tok.vocab_size, "itos": tok.itos, "stoi": tok.stoi}, f)
    elif tipo == "gpt2":
        tok = TokTiktoken("gpt2")
        info_tok = {"tipo": "tiktoken", "nome": "gpt2"}
    else:
        tok = TokHF(HF_PTBR)
        info_tok = {"tipo": "hf", "nome": HF_PTBR}
    assert tok.vocab_size < 2 ** 16, "os ids precisam caber em uint16"

    info = {"tokenizador": info_tok, "vocab_size": tok.vocab_size}
    for nome, texto in textos.items():
        ids = tok.encode(texto)
        # a avaliação em bits por caractere depende de a tokenização ser exata
        assert tok.decode(ids) == texto, f"{tipo}: decode(encode(x)) != x em {nome}"
        np.array(ids, dtype=np.uint16).tofile(destino / f"{nome}.bin")
        info[nome] = {"tokens": len(ids), "caracteres": len(texto),
                      "caracteres_por_token": round(len(texto) / len(ids), 4)}
        print(f"[{tipo}] {nome}: {len(texto):,} caracteres -> {len(ids):,} tokens")

    (destino / "tokenizador.json").write_text(json.dumps(info_tok, ensure_ascii=False))
    (destino / "info.json").write_text(json.dumps(info, ensure_ascii=False, indent=2))
    return info


def main():
    ap = argparse.ArgumentParser(description="Gera os .bin do nanoGPT para cada tokenizador.")
    ap.add_argument("tipos", nargs="+", choices=sorted(DATASETS))
    args = ap.parse_args()
    for t in args.tipos:
        prepara(t)


if __name__ == "__main__":
    sys.exit(main())
