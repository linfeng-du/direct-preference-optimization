accelerate launch src/train.py \
    exp_name=gpt2-mixture \
    dataset=persona \
    dataset.prepend_persona=false \
    dataset.n_clusters=8 \
    model=gpt2-instruct \
    adapter=mixture_of_loras \
    adapter.n_loras=8 \
    adapter.r=2 \
    batch_size=8 \
    eval_batch_size=32
