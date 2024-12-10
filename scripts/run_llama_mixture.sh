accelerate launch src/train.py \
    model=llama1B-instruct \
    dataset.prepend_persona=false \
    adapter=mixture_of_loras \
    dataset.n_clusters=8 \
    exp_name=llama1B-mixture
