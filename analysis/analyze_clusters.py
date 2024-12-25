import sys

import datasets

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans

sys.path.append('src')
from preference_datasets.persona import _get_split_indices, _extract_raw_features


def main():
    feature_set = [
        'age', 'sex', 'race', 'ancestry', 'household language',
        'education', 'employment status', 'marital status',
        'household type', 'family presence and age', 'place of birth',
        'citizenship', 'disability', 'health insurance',
        'Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism',
        'religion'
    ]
    dataset = datasets.load_dataset('SynthLabsAI/PERSONA', split='train')
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    raw_features = np.array([
        _extract_raw_features(dataset[i]['persona'])
        for i in range(len(dataset))
    ])
    numerical_features = raw_features[:, [0]].astype(np.float64)
    categorical_features = raw_features[:, 1:]

    encoder.fit(categorical_features[_get_split_indices(split='train')])
    feature_dims = [len(c) for c in encoder.categories_]

    categorical_features = encoder.transform(categorical_features)
    data = np.hstack((numerical_features, categorical_features))

    for n_clusters in [2, 4, 8, 16]:
        kmeans = KMeans(n_clusters, n_init='auto', random_state=42)
        kmeans.fit(data[_get_split_indices(split='train')])

        print('=' * 40, f'n_clusters={n_clusters}', '=' * 40)
        for i, cluster_center in enumerate(kmeans.cluster_centers_):
            print('=' * 20, f'cluster {i}', '=' * 20)
            print(f'{feature_set[0]}: {round(cluster_center[0])}')
            cluster_center = cluster_center[1:]

            for i, feature_dim in enumerate(feature_dims):
                value = encoder.categories_[i][np.argmax(cluster_center[:feature_dim])]
                print(f'{feature_set[i+1]}: {value}') 
                cluster_center = cluster_center[feature_dim:]


if __name__ == '__main__':
    main()
