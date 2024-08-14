accelerate launch train_accelerate.py train_reward=True model=gpt-2-instruct trainer=BasicTrainer n_epochs=1 use_hnet=True lr=5e-5 wandb.enabled=false exp_name=reward_fun_base

