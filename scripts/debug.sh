accelerate launch src/train.py \
    exp_name=debug \
    dataset=persona \
    dataset.prepend_persona=false \
    dataset.n_clusters=8 \
    model=gpt2-instruct \
    adapter=mixture_of_loras \
    adapter.n_loras=8 \
    adapter.r=2 \
    n_examples=256 \
    n_eval_examples=256
