accelerate launch train.py \
    model=gpt-2-instruct \
    trainer=BasicTrainer \
    n_epochs=1 \
    use_hnet=false \
    lr=5e-5 \
    wandb.enabled=true \
    use_lora=true \
    exp_name=lora_dpo

