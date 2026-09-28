"""Avaliação comparável entre tokenizadores: bits por caractere (bpc).

Autoria: código original do aluno. Usa a classe `GPT` do nanoGPT
(`nanoGPT-master/model.py`, Karpathy, MIT) apenas para carregar os modelos.

A perda de validação do `train.py` é média por token. Um token de caractere e
um token BPE não carregam a mesma quantidade de texto, então essas perdas não
se comparam diretamente. Aqui a perda total (em nats) sobre o conjunto de teste
inteiro é dividida pelo número de caracteres e convertida para bits:

    bpc = soma_das_NLL / (n_caracteres * ln 2)

Também se reporta a perplexidade por palavra, exp(soma_das_NLL / n_palavras),
que tem a mesma propriedade de independer do tokenizador.

Cada token do teste é pontuado exatamente uma vez, com janela deslizante: a
primeira janela pontua todas as posições e as seguintes avançam `passo` tokens
(metade do contexto, por padrão) e pontuam só as posições novas. Assim todo
token, exceto os da primeira janela, é previsto com pelo menos meio contexto.

Uso:
    python -m machado_gpt.avaliacao --run runs/char_baby
    python -m machado_gpt.avaliacao --hf gpt2 --dataset machado_gpt2
    python -m machado_gpt.avaliacao --run runs/ptbr_ft --limite-chars 20000   # estimativa rápida
"""

import argparse
import json
import math
import re
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

RAIZ = Path(__file__).resolve().parent.parent
NANOGPT = RAIZ / "nanoGPT-master"
CORPUS = RAIZ / "corpus"
sys.path.insert(0, str(NANOGPT))

from model import GPT, GPTConfig  # noqa: E402  (nanoGPT, Karpathy)

from machado_gpt import tokenizacao  # noqa: E402


class Modelo:
    """Envelope comum para modelos do nanoGPT, do vídeo e checkpoints do HF.

    `logits(x)` devolve os logits de todas as posições, (B, T, V).
    """

    def __init__(self, rede, block_size, dataset, tipo, nome):
        self.rede, self.block_size, self.dataset = rede, block_size, dataset
        self.tipo, self.nome = tipo, nome

    def logits(self, x):
        if self.tipo == "video":
            return self.rede(x)[0]
        # com targets, o forward do nanoGPT calcula lm_head em todas as posições
        return self.rede(x, torch.zeros_like(x))[0]

    @torch.no_grad()
    def gera(self, idx, max_new_tokens, temperature=0.8, top_k=200):
        if self.tipo == "video":
            from machado_gpt import video_gpt
            return video_gpt.gera_com_amostragem(self.rede, idx, max_new_tokens, temperature, top_k)
        return self.rede.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k)

    def n_params(self):
        return sum(p.numel() for p in self.rede.parameters())


def carrega_run(run_dir, device="cpu"):
    """Carrega o checkpoint de um run (nanoGPT ou vídeo)."""
    run_dir = Path(run_dir)
    ck_path = run_dir / "ckpt.pt"
    if not ck_path.exists():
        ck_path = run_dir / "ckpt_slim.pt"
    ck = torch.load(ck_path, map_location=device, weights_only=False)
    if ck.get("tipo") == "video":
        from machado_gpt import video_gpt
        video_gpt.configura(**ck["hparams"], device=device)
        rede = video_gpt.GPTLanguageModel()
        rede.load_state_dict({k: v.float() for k, v in ck["model"].items()})
        rede.to(device).eval()
        return Modelo(rede, ck["hparams"]["block_size"], ck["dataset"], "video", run_dir.name)
    rede = GPT(GPTConfig(**ck["model_args"]))
    sd = {k.removeprefix("_orig_mod."): v.float() for k, v in ck["model"].items()}
    rede.load_state_dict(sd)
    rede.to(device).eval()
    return Modelo(rede, ck["model_args"]["block_size"], ck["config"]["dataset"], "nanogpt", run_dir.name)


def carrega_hf(hf_id, dataset, block_size=1024, device="cpu"):
    rede = GPT.from_pretrained(hf_id, dict(dropout=0.0))
    if block_size < rede.config.block_size:
        rede.crop_block_size(block_size)
    rede.to(device).eval()
    return Modelo(rede, rede.config.block_size, dataset, "nanogpt", hf_id)


@torch.no_grad()
def nll_total(modelo, ids, device="cpu", passo=None, lote=16, ctx=nullcontext()):
    """Soma das NLL (nats) de ids[1:], cada token pontuado uma vez."""
    T = modelo.block_size
    passo = passo or T // 2
    ids = torch.as_tensor(np.asarray(ids, dtype=np.int64))
    N = len(ids)
    janelas = []  # (inicio, fim, quantos alvos novos no fim da janela)
    fim_anterior = 0
    fim = min(T, N - 1)
    while True:
        inicio = max(0, fim - T)
        janelas.append((inicio, fim, fim - fim_anterior))
        fim_anterior = fim
        if fim == N - 1:
            break
        fim = min(fim + passo, N - 1)

    soma, contagem = 0.0, 0
    for i in range(0, len(janelas), lote):
        grupo = janelas[i:i + lote]
        comp = max(f - s for s, f, _ in grupo)
        x = torch.zeros((len(grupo), comp), dtype=torch.long)
        y = torch.full((len(grupo), comp), -1, dtype=torch.long)
        for b, (s, f, novos) in enumerate(grupo):
            n = f - s
            x[b, :n] = ids[s:f]
            y[b, n - novos:n] = ids[f - novos + 1:f + 1]
        x, y = x.to(device), y.to(device)
        with ctx:
            logits = modelo.logits(x)
        perdas = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1),
                                 ignore_index=-1, reduction="sum")
        soma += perdas.item()
        contagem += int((y >= 0).sum().item())
    assert contagem == N - 1
    return soma, contagem


def avalia_texto(modelo, tok, texto, device="cpu", ctx=nullcontext(), lote=16):
    ids = tok.encode(texto)
    t0 = time.time()
    soma, n = nll_total(modelo, ids, device=device, ctx=ctx, lote=lote)
    # caracteres efetivamente previstos (tudo menos o primeiro token)
    n_chars = len(texto) - len(tok.decode(ids[:1]))
    n_palavras = len(re.findall(r"\w+", texto))
    return {
        "tokens": n,
        "caracteres": n_chars,
        "perda_por_token": soma / n,
        "bpc": soma / (n_chars * math.log(2)),
        "ppl_palavra": math.exp(soma / n_palavras),
        "segundos": round(time.time() - t0, 1),
    }


def avalia(modelo, device="cpu", splits=("val", "test"), por_genero=True, lote=16, limite_chars=None):
    """Avalia em val/test e no teste por gênero. `limite_chars` corta os textos (teste rápido)."""
    data_dir = NANOGPT / "data" / modelo.dataset
    tok = tokenizacao.carrega(data_dir)
    ctx = (torch.amp.autocast(device_type="cuda", dtype=torch.float16)
           if "cuda" in str(device) else nullcontext())
    res = {"modelo": modelo.nome, "dataset": modelo.dataset, "parametros": modelo.n_params(),
           "block_size": modelo.block_size}
    for s in splits:
        texto = (CORPUS / f"{s}.txt").read_text(encoding="utf-8")[:limite_chars]
        res[s] = avalia_texto(modelo, tok, texto, device, ctx, lote)
        print(f"[{modelo.nome}] {s}: bpc={res[s]['bpc']:.4f} ppl_palavra={res[s]['ppl_palavra']:.1f}",
              flush=True)
    if por_genero:
        res["test_por_genero"] = {}
        for arq in sorted((CORPUS / "test_por_genero").glob("*.txt")):
            texto = arq.read_text(encoding="utf-8")[:limite_chars]
            r = avalia_texto(modelo, tok, texto, device, ctx, lote)
            res["test_por_genero"][arq.stem] = {"bpc": r["bpc"], "caracteres": r["caracteres"]}
    return res


def main():
    ap = argparse.ArgumentParser(description="Avalia um modelo em bits por caractere.")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", help="diretório do run com ckpt.pt")
    g.add_argument("--hf", help="id de checkpoint GPT-2 do Hugging Face (zero-shot)")
    ap.add_argument("--dataset", help="dataset do tokenizador (obrigatório com --hf)")
    ap.add_argument("--block-size", type=int, default=1024)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--lote", type=int, default=16)
    ap.add_argument("--saida", help="json de saída")
    ap.add_argument("--limite-chars", type=int, default=None,
                    help="avalia só os primeiros N caracteres de cada texto (estimativa rápida)")
    args = ap.parse_args()

    if args.run:
        modelo = carrega_run(args.run, args.device)
    else:
        modelo = carrega_hf(args.hf, args.dataset, args.block_size, args.device)
    res = avalia(modelo, args.device, lote=args.lote, limite_chars=args.limite_chars)
    txt = json.dumps(res, ensure_ascii=False, indent=2)
    if args.saida:
        Path(args.saida).parent.mkdir(parents=True, exist_ok=True)
        Path(args.saida).write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
