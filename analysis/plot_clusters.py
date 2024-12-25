import sys

import datasets

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import OneHotEncoder
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

sys.path.append('src')
from preference_datasets.persona import _get_split_indices, _extract_raw_features


def plot_clusters(data, reduced_data, n_clusters):
    kmeans = KMeans(n_clusters, n_init='auto', random_state=42)
    kmeans_2d = KMeans(n_clusters, n_init='auto', random_state=42)

    kmeans.fit(data[_get_split_indices(split='train')])
    kmeans_2d.fit(reduced_data[_get_split_indices(split='train')])
    cluster_centers = kmeans_2d.cluster_centers_

    colors = np.empty(len(reduced_data), dtype=np.str_)
    colors[_get_split_indices(split='train')] = 'k'
    colors[_get_split_indices(split='val')] = 'k'
    colors[_get_split_indices(split='test')] = 'k'
    colors[_get_split_indices(split='test_unseen')] = 'r'
    colors = colors.tolist()

    x_min, x_max = reduced_data[:, 0].min() - 1, reduced_data[:, 0].max() + 1
    y_min, y_max = reduced_data[:, 1].min() - 1, reduced_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))
    extent = (xx.min(), xx.max(), yy.min(), yy.max())

    Z = kmeans_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.imshow(Z, aspect='auto', interpolation='nearest', origin='lower', extent=extent)
    plt.scatter(reduced_data[:, 0], reduced_data[:, 1], s=1, c=colors)
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=20, marker='x', color='w')

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xticks([])
    plt.yticks([])


def main():
    dataset = datasets.load_dataset('SynthLabsAI/PERSONA', split='train')

    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    tsne = TSNE(n_components=2, random_state=42)

    raw_features = np.array([
        _extract_raw_features(dataset[i]['persona'])
        for i in range(len(dataset))
    ])
    numerical_features = raw_features[:, [0]].astype(np.float64)
    categorical_features = raw_features[:, 1:]

    encoder.fit(categorical_features[_get_split_indices(split='train')])
    categorical_features = encoder.transform(categorical_features)

    data = np.hstack((numerical_features, categorical_features))
    reduced_data = tsne.fit_transform(data).astype(np.float64)

    plt.figure(figsize=(4, 3))
    plot_clusters(data, reduced_data, n_clusters=8)
    plt.tight_layout()
    plt.savefig('./analysis/figures/clusters.png', dpi=300, bbox_inches='tight')


if __name__ == '__main__':
    main()
