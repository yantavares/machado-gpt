# Ajuste fino do GPT-2 small (124M, OpenAI, pré-treinado em inglês) no corpus
# (autoria do aluno). Segue config/finetune_shakespeare.py do nanoGPT, com
# taxa menor, decaimento cosseno e contexto de 512 tokens.
exec(open('../experiments/configs/_comum.py').read())
out_dir = '../runs/gpt2_ft'
dataset = 'machado_gpt2'
init_from = 'gpt2'
eval_interval = 100
eval_iters = 40
log_interval = 10
# 8 x 4 x 512 = 16.384 tokens por iteração; 1 época ~ 290 iterações
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
