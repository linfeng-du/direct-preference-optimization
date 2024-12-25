import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.append('src')
from preference_datasets import load_persona


def get_all_proximities(n_clusters, sparse_proximities):
    persona_proximities = {
        persona: proximities
        for prompt_data in load_persona(
            split='train',
            prepend_persona=False,
            n_clusters=n_clusters,
            sparse_proximities=sparse_proximities
        ).values()
        for persona, proximities in zip(prompt_data['persona'], prompt_data['proximities'])
    }
    persona_proximities.update({
        persona: proximities
        for prompt_data in load_persona(
            split='test_unseen',
            prepend_persona=False,
            n_clusters=n_clusters,
            sparse_proximities=sparse_proximities
        ).values()
        for persona, proximities in zip(prompt_data['persona'], prompt_data['proximities'])
    })
    return np.array(list(persona_proximities.values()))


def main():
    proximities = get_all_proximities(n_clusters=8, sparse_proximities=False)
    sparse_proximities = get_all_proximities(n_clusters=8, sparse_proximities=True)

    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6), constrained_layout=True)

    im0 = axes[0].imshow(proximities, vmin=0, vmax=1, aspect='auto', origin='lower')
    axes[0].set_ylabel('Persona', fontsize=12)

    axes[1].imshow(sparse_proximities, vmin=0, vmax=1, aspect='auto', origin='lower')
    axes[1].set_yticks([])

    fig.supxlabel('Cluster', fontsize=12)
    fig.colorbar(im0, ax=axes, orientation='vertical', fraction=0.04, pad=0.02)

    plt.savefig('./analysis/figures/proximities.png', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()
