# GPT de Machado de Assis

Projeto Individual 1 de Redes Neurais e Aprendizado Profundo (UnB, 2026/2, Prof. Díbio):

implementação, treino e avaliação de modelos de linguagem no estilo do GPT-2 sobre a obra completa de Machado de Assis, com o [nanoGPT](https://github.com/karpathy/nanoGPT) como base. O enunciado está em [projeto1-rnap](projeto1-rnap.pdf).

## Estrutura

- `machado/`: acervo original (machado.mec.gov.br, igual ao corpus `machado` do NLTK), em Windows-1252.
- `archive/`: segunda edição digital das obras, usada só para validar a limpeza.
- `machado_gpt/corpus.py`: limpeza, deduplicação e divisão treino/validação/teste.
- `machado_gpt/tokenizacao.py`: caractere, BPE do GPT-2 e BPE do GPorTuguese-2 para os `.bin` do nanoGPT.
- `machado_gpt/avaliacao.py`: bits por caractere no teste, comparáveis entre tokenizadores.
- `machado_gpt/geracao.py`: amostras e métricas do texto gerado.
- `machado_gpt/baselines.py`: n-gramas de caracteres.
- `machado_gpt/rag.py`: protótipo de perguntas e respostas com recuperação de passagens.
- `machado_gpt/graficos.py`: figuras e tabelas do artigo a partir de `results/`.
- `experiments/configs/`: configurações do nanoGPT; `experiments/executar.py`: treino, avaliação e amostras de cada modelo; `experiments/colab.sh`: execução numa GPU do Colab.
- `results/`: métricas, amostras e tabelas de todos os experimentos.

## Reproduzir

```sh
pip install torch numpy tiktoken transformers scikit-learn matplotlib
python -m machado_gpt.corpus
python -m machado_gpt.tokenizacao char gpt2 ptbr
python -m machado_gpt.baselines
python experiments/executar.py --todos          # em GPU; ou um run por vez, ex.: char_baby
python -m machado_gpt.rag --avaliar
python -m machado_gpt.graficos
cd paper && make
```

Ou abra o notebook no Colab com GPU e execute todas as células.
