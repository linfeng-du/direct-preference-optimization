while [[ "$#" -gt 0 ]]; do
    case $1 in
        --n_clusters) n_clusters="$2"; shift ;;
        --sparse_proximities) sparse_proximities="$2"; shift ;;
    esac
    shift
done

r=$((16 / $n_clusters))

accelerate launch src/train.py \
    exp_name=llama-mixture/n_clusters=$n_clusters-sparse_proximities=$sparse_proximities \
    dataset=persona \
    dataset.prepend_persona=false \
    dataset.n_clusters=8 \
    model=llama1B-instruct \
    adapter=mixture_of_loras \
    adapter.n_loras=8 \
    adapter.r=2 \
    batch_size=4 \
    eval_every=40000 \
    eval_batch_size=8
