# Char-level maior (autoria do aluno): 8 camadas, 512 dimensões, contexto de
# 512 caracteres (~25M parâmetros). Mesmo número de tokens por iteração do
# baby GPT (32 x 512 = 64 x 256).
exec(open('../experiments/configs/_comum.py').read())
out_dir = '../runs/char_large'
dataset = 'machado_char'
eval_interval = 250
eval_iters = 100
gradient_accumulation_steps = 1
batch_size = 32
block_size = 512
n_layer = 8
n_head = 8
n_embd = 512
dropout = 0.2
learning_rate = 6e-4
max_iters = 6000
lr_decay_iters = 6000
min_lr = 6e-5
beta2 = 0.99
warmup_iters = 200
