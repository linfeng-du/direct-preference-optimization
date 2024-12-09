accelerate launch src/train.py \
    exp_name=llama-baseline \
    dataset=persona \
    dataset.prepend_persona=false \
    dataset.n_clusters=null \
    model=llama1B-instruct \
    adapter=lora \
    adapter.r=16 \
    batch_size=16 \
    eval_every=39608 \
    eval_batch_size=32
