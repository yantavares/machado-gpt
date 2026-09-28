"""Testes do gerador do notebook (autoria do aluno). Rodar: python -m pytest tests/"""
import ast
import subprocess
import sys
from pathlib import Path

import pytest

RAIZ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(RAIZ / "notebooks"))
import construir_notebook as cn  # noqa: E402

PRISTINO = "4e56ee5"  # commit com o nanoGPT sem alterações


def sem_docstrings(fonte):
    arvore = ast.parse(fonte)
    for no, d in list(cn._docstrings(arvore)):
        if isinstance(no, ast.Module):
            no.body.remove(d)
        else:
            d.value.value = ""
    return ast.dump(arvore)


def celulas_writefile():
    for celula in cn.construir(cn.TODOS).cells:
        primeira, _, resto = celula.source.partition("\n")
        if primeira.startswith("%%writefile "):
            yield primeira.split()[1], resto


def test_notebook_roda_o_mesmo_codigo_do_repositorio():
    arquivos = dict(celulas_writefile())
    assert "machado_gpt/corpus.py" in arquivos
    for rel, fonte in arquivos.items():
        assert sem_docstrings(fonte) == sem_docstrings((RAIZ / rel).read_text(encoding="utf-8")), rel


def test_enxuga_preserva_marcas_de_autoria():
    fonte = (RAIZ / "machado_gpt" / "video_gpt.py").read_text(encoding="utf-8")
    codigo = fonte.split('"""', 2)[2]  # sem o docstring de módulo
    assert cn.enxuga(fonte, comentarios=False, modulo=None).count("[ALUNO]") == codigo.count("[ALUNO]")
    assert "Autoria" in cn.enxuga((RAIZ / "machado_gpt" / "corpus.py").read_text(encoding="utf-8"))


def test_celula_do_nanogpt_reproduz_as_modificacoes(tmp_path):
    try:
        for arq in ("model.py", "train.py"):
            original = subprocess.run(["git", "show", f"{PRISTINO}:nanoGPT-master/{arq}"], cwd=RAIZ,
                                      capture_output=True, text=True, check=True).stdout
            (tmp_path / "nanoGPT-master").mkdir(exist_ok=True)
            (tmp_path / "nanoGPT-master" / arq).write_text(original, encoding="utf-8")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pytest.skip("histórico do git indisponível")
    celula = cn.celula_nanogpt().source
    python = "\n".join(l for l in celula.splitlines() if not l.startswith("!"))
    import os
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        exec(python, {})
    finally:
        os.chdir(cwd)
    for arq in ("model.py", "train.py"):
        gerado = (tmp_path / "nanoGPT-master" / arq).read_text(encoding="utf-8")
        repo = (RAIZ / "nanoGPT-master" / arq).read_text(encoding="utf-8")
        assert ast.dump(ast.parse(gerado)) == ast.dump(ast.parse(repo)), arq
        assert gerado.count(">>> MODIFICAÇÃO DO ALUNO") == repo.count(">>> MODIFICAÇÃO DO ALUNO")
