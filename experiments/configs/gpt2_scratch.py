# BPE do zero (autoria do aluno): mesma arquitetura do baby GPT (6/6/384,
# contexto de 256 tokens), mas com o tokenizador BPE 'gpt2' (vocabulário de
# 50.257). Isola o efeito do tokenizador, sem pré-treino.
exec(open('../experiments/configs/_comum.py').read())
out_dir = '../runs/gpt2_scratch'
dataset = 'machado_gpt2'
eval_interval = 250
eval_iters = 100
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
learning_rate = 1e-3
max_iters = 3000
lr_decay_iters = 3000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100
