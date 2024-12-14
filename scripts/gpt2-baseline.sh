accelerate launch src/train.py \
    exp_name=gpt2-baseline \
    dataset=persona \
    model=gpt2-instruct \
    adapter=lora \
    adapter.r=16 \
    batch_size=16 \
    eval_every=20000 \
    eval_batch_size=32
