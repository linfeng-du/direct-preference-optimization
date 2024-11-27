# python -u train.py model=gpt-2-instruct trainer=FSDPHnetTrainer use_hnet=true hnet.use_dummies=True wandb=true


# Hyperparameter sweep
for hnet_a in 16 32; do
        for lr in 5e-2 5e-3; do
            accelerate launch train.py model=gpt-2-instruct trainer=BasicTrainer lr=$lr n_epochs=1 wandb.enabled=true use_hnet=true exp_name=hnet_dpo_hnet_a_dummies_${hnet_a}_$lr
        done
    done
done

# accelerate launch train.py model=gpt-2-instruct trainer=BasicTrainer  hnet=base_net lr=5e-2 n_epochs=1 wandb.enabled=true use_hnet=true exp_name=hnet_dpo
