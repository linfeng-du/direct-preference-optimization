accelerate launch src/train.py \
    exp_name=llama-persona \
    dataset=persona \
    dataset.prepend_persona=true \
    dataset.n_clusters=null \
    model=llama1B-instruct \
    adapter=lora \
    adapter.r=16 \
    batch_size=8 \
    eval_batch_size=64
