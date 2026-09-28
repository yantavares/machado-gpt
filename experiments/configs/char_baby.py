# "Baby GPT" char-level (autoria do aluno): hiperparâmetros idênticos aos de
# config/train_shakespeare_char.py do nanoGPT e aos do gpt.py do vídeo
# (6 camadas, 6 cabeças, 384 dimensões, contexto de 256 caracteres).
exec(open('../experiments/configs/_comum.py').read())
out_dir = '../runs/char_baby'
dataset = 'machado_char'
eval_interval = 250
eval_iters = 200
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256
n_layer = 6
n_head = 6
n_embd = 384
dropout = 0.2
learning_rate = 1e-3
max_iters = 5000
lr_decay_iters = 5000
min_lr = 1e-4
beta2 = 0.99
warmup_iters = 100
