"""Testes do cálculo de bpc (autoria do aluno). Rodar: python -m pytest tests/"""
import math

import numpy as np
import torch
from torch.nn import functional as F

from machado_gpt.avaliacao import GPT, GPTConfig, Modelo, nll_total
from machado_gpt.corpus import chave_indice, remove_indices
from machado_gpt.tokenizacao import TokChar


def modelo_pequeno(T=16, V=11):
    torch.manual_seed(0)
    rede = GPT(GPTConfig(n_layer=1, n_head=2, n_embd=16, block_size=T, vocab_size=V, dropout=0.0))
    return Modelo(rede.eval(), T, "x", "nanogpt", "teste")


def nll_ingenua(modelo, ids, passo):
    """Recalcula token a token: alvo p usa a janela que o termina no mesmo lugar."""
    T = modelo.block_size
    N = len(ids)
    total, fim_ant, fim = 0.0, 0, min(T, N - 1)
    while True:
        ini = max(0, fim - T)
        x = torch.tensor([ids[ini:fim]])
        lg = modelo.logits(x)[0]
        for p in range(fim_ant + 1, fim + 1):
            total += F.cross_entropy(lg[p - 1 - ini], torch.tensor(ids[p])).item()
        fim_ant = fim
        if fim == N - 1:
            break
        fim = min(fim + passo, N - 1)
    return total


def test_cada_token_uma_vez_e_bate_com_calculo_ingenuo():
    m = modelo_pequeno()
    ids = np.random.default_rng(0).integers(0, 11, size=103).tolist()
    for passo in (1, 5, 8, 16):
        soma, n = nll_total(m, ids, passo=passo, lote=3)
        assert n == len(ids) - 1
        assert math.isclose(soma, nll_ingenua(m, ids, passo), rel_tol=1e-4)


def test_texto_menor_que_o_contexto():
    m = modelo_pequeno(T=64)
    ids = list(range(10))
    soma, n = nll_total(m, ids)
    assert n == 9 and soma > 0


def test_tokenizador_char_ida_e_volta():
    tok = TokChar.do_texto("Capitu — olhos de ressaca.")
    assert tok.decode(tok.encode("olhos de ressaca")) == "olhos de ressaca"


def test_indice_removido_ate_a_repeticao():
    pars = ["Título", "ÍNDICE", "CAPÍTULO PRIMEIRO", "CAPÍTULO II",
            "CAPÍTULO PRIMEIRO", "Era uma vez " * 20, "CAPÍTULO II", "Fim."]
    log = []
    assert remove_indices(pars, log) == ["Título", "CAPÍTULO PRIMEIRO", "Era uma vez " * 20,
                                         "CAPÍTULO II", "Fim."]
    assert log[0]["repeticao"]


def test_indice_com_subtitulo():
    pars = ["ÍNDICE", "CAPÍTULO PRIMEIRO - TRÊS AMIGOS", "CAPÍTULO II - O PONTO",
            "CAPÍTULO PRIMEIRO", "TRÊS AMIGOS", "Texto " * 40]
    assert remove_indices(pars, [])[0] == "CAPÍTULO PRIMEIRO"
    assert chave_indice("Índice.") == "ÍNDICE"
