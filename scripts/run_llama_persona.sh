accelerate launch src/train.py \
    model=llama1B-instruct \
    prepend_persona=true \
    exp_name=llama1B-persona
