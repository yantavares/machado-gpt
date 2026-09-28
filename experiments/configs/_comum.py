# Configuração comum a todos os experimentos (autoria do aluno).
# Os arquivos de config são executados pelo configurator.py do nanoGPT
# (exec), a partir de nanoGPT-master/; por isso out_dir aponta para ../runs/.
# GPU T4 (Colab): sem bfloat16 nativo, então float16 com GradScaler.
device = 'cuda'
dtype = 'float16'
compile = True
wandb_log = False
always_save_checkpoint = False  # só salva quando a validação melhora
log_interval = 50
