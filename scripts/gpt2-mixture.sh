while [[ "$#" -gt 0 ]]; do
    case $1 in
        --n_clusters) n_clusters="$2"; shift ;;
        --sparse_proximities) sparse_proximities="$2"; shift ;;
    esac
    shift
done

r=$((16 / $n_clusters))

accelerate launch src/train.py \
    exp_name=gpt2-mixture/n_clusters=$n_clusters-sparse_proximities=$sparse_proximities \
    dataset=persona \
    dataset.prepend_persona=false \
    dataset.n_clusters=$n_clusters \
    model=gpt2-instruct \
    adapter=mixture_of_loras \
    adapter.n_loras=$n_clusters \
    adapter.r=$r \
    batch_size=16 \
    eval_every=20000 \
    eval_batch_size=32
