accelerate launch train_accelerate.py model=gpt-2-instruct trainer=BasicTrainer use_hnet=false lr=5e-6 wandb.enabled=true exp_name=dpo

