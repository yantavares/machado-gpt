# Ajuste fino do GPorTuguese-2 (124M, GPT-2 small adaptado à Wikipédia em português por P. Guillou) no corpus
# (autoria do aluno). Segue config/finetune_shakespeare.py do nanoGPT, com
# taxa menor, decaimento cosseno e contexto de 512 tokens.
exec(open('../experiments/configs/_comum.py').read())
out_dir = '../runs/ptbr_ft'
dataset = 'machado_ptbr'
init_from = 'pierreguillou/gpt2-small-portuguese'
eval_interval = 100
eval_iters = 40
log_interval = 10
# 8 x 4 x 512 = 16.384 tokens por iteração; 1 época ~ 183 iterações
batch_size = 8
gradient_accumulation_steps = 4
block_size = 512
dropout = 0.1
learning_rate = 6e-5
max_iters = 1200
lr_decay_iters = 1200
min_lr = 6e-6
warmup_iters = 50
weight_decay = 0.1
