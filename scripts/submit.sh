#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --gpus-per-node=2
#SBATCH --account=def-cpsmcgil

source /home/linfeng/.bashrc
nvidia-smi

activate torch

for seed in {0..2}; do
	bash scripts/llama1B-baseline.sh --seed $seed
	bash scripts/llama1B-persona.sh --seed $seed
	bash scripts/llama1B-mixture.sh \
		--seed $seed \
		--n_clusters 8 \
		--sparse_proximities false
done

for seed in {0..2}; do
	bash scripts/llama1B-mixture.sh \
		--seed $seed \
		--n_clusters 8 \
		--sparse_proximities true

	for n_clusters in {2,4,16}; do
		bash scripts/llama1B-mixture.sh \
			--seed $seed \
			--n_clusters $n_clusters \
			--sparse_proximities false

		bash scripts/llama1B-mixture.sh \
			--seed $seed \
			--n_clusters $n_clusters \
			--sparse_proximities true
	done
done
