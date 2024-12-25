#!/bin/bash

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --seed) seed="$2"; shift ;;
    esac
    shift
done

accelerate launch src/train.py \
    seed=$seed \
    exp_name=llama1B-context \
    dataset=persona \
    dataset.prepend_persona=true \
    model=llama1B-instruct \
    adapter=lora \
    adapter.r=16
