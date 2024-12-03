accelerate launch train.py \
    model=gpt-2-instruct \
    trainer=BasicTrainer \
    hnet=lora_hnet \
    use_hnet=true \
    lr=5e-6 \
    wandb.enabled=true \
    exp_name=dpo \
    hnet_type=lora_hnet \
    exp_name=lora_hnet
