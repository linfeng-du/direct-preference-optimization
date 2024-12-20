import random
from collections import defaultdict

import datasets

import numpy as np
import scipy.sparse
import scipy.special
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans

from tqdm import tqdm


def load_persona(
    split: str,
    prepend_persona: bool,
    n_clusters: int | None = None,
    sparse_proximities: bool | None = None
) -> dict[str, dict[str, list[str] | list[float] | list[tuple[int, int]]]]:
    """Load the PERSONA dataset from Hugging Face and convert it to the necessary format.

    The dataset is converted to a dictionary with the following structure:
    {
        'prompt1': {
            'responses': list[str],
            'pairs': list[tuple[int, int]],
            'persona': list[str],
            [Optional] 'proximities': list[float]
        },
        ...
    }

    Prompts are structured as follows:
        \n\nHuman: <prompt>\n\nAssistant:
    """
    dataset = datasets.load_dataset('SynthLabsAI/PERSONA', split='train')
    data = defaultdict(lambda: defaultdict(list))

    if n_clusters is not None:
        split_proximities = _get_split_proximities(dataset, split, n_clusters, sparse_proximities)

    for split_index, example_index in enumerate(
        tqdm(_get_split_indices(split), desc=f'Processing PERSONA {split} split')
    ):
        example = dataset[example_index]

        prompt = example['instruction']
        chosen = example['data']
        rejected = example['original']
        persona = example['persona']

        if prepend_persona:
            persona_row = ', '.join(persona.strip().split('\n'))
            prompt = f'Here are your characteristics: {persona_row}. {prompt}'

        prompt = f'\n\nHuman: {prompt}\n\nAssistant:'
        responses = [f' {chosen}', f' {rejected}']
        n_responses = len(data[prompt]['responses'])

        data[prompt]['responses'].extend(responses)
        data[prompt]['pairs'].append((n_responses, n_responses + 1))
        data[prompt]['persona'].append(persona)

        if n_clusters is not None:
            proximities = split_proximities[split_index]
            data[prompt]['proximities'].append(proximities)

    return data


def _get_split_indices(split: str) -> list[int]:
    """Get the indices of examples from the specified split of the PERSONA dataset."""
    if not hasattr(_get_split_indices, '_cache'):
        # Create training, validation, test, and unseen test splits
        # The dataset contains 1,000 personas, each with 200 response pairs
        generator = random.Random(x=42)

        # Randomly set aside 100 personas as the unseen test set
        unseen_personas = generator.sample(range(1000), k=100)
        seen_personas = [p for p in range(1000) if p not in unseen_personas]

        test_unseen_indices = [
            index
            for p in unseen_personas
            for index in range(p * 200, (p + 1) * 200)
        ]

        # For each of the remaining 900 personas, randomly split its 200 responses
        # into training, validation, and test sets in a 8:1:1 ratio
        train_indices = []
        validation_indices = []
        test_indices = []

        for persona in seen_personas:
            indices = list(range(persona * 200, (persona + 1) * 200))
            generator.shuffle(indices)

            train_indices.extend(indices[:160])
            validation_indices.extend(indices[160:180])
            test_indices.extend(indices[180:])

        cache = {
            'train': train_indices,
            'validation': validation_indices,
            'test': test_indices,
            'test_unseen': test_unseen_indices
        }
        setattr(_get_split_indices, '_cache', cache)

    return getattr(_get_split_indices, '_cache')[split]


def _get_split_proximities(
    dataset: datasets.Dataset,
    split: str,
    n_clusters: int,
    sparse_proximities: bool
) -> list[list[float]]:
    """Get the cluster proximities of examples from the specified split of the PERSONA dataset."""
    cache_key = f'_cache_{(n_clusters, sparse_proximities)}'

    if not hasattr(_get_split_proximities, cache_key):
        encoder = OneHotEncoder(handle_unknown='ignore')
        kmeans = KMeans(n_clusters, n_init='auto', random_state=42)

        def compute_split_proximities(split):
            raw_features = np.array([
                _extract_raw_features(dataset[index]['persona'])
                for index in _get_split_indices(split)
            ])

            numerical_features = raw_features[:, [0]].astype(np.float64)
            categorical_features = raw_features[:, 1:]

            if split == 'train':
                categorical_features = encoder.fit_transform(categorical_features)
                features = scipy.sparse.hstack((numerical_features, categorical_features))
                distances = kmeans.fit_transform(features)
            else:
                categorical_features = encoder.transform(categorical_features)
                features = scipy.sparse.hstack((numerical_features, categorical_features))
                distances = kmeans.transform(features)

            if sparse_proximities:
                proximities = np.zeros_like(distances)
                proximities[np.arange(distances.shape[0]), np.argmin(distances, axis=-1)] = 1.
            else:
                proximities = scipy.special.softmax(-np.log(distances), axis=-1)

            return proximities.tolist()

        cache = {
            'train': compute_split_proximities(split='train'),
            'validation': compute_split_proximities(split='validation'),
            'test': compute_split_proximities(split='test'),
            'test_unseen': compute_split_proximities(split='test_unseen')
        }
        setattr(_get_split_proximities, cache_key, cache)

    return getattr(_get_split_proximities, cache_key)[split]


def _extract_raw_features(persona: str) -> list[str | int]:
    """Extract raw features from the persona."""
    feature_set = {
        'age', 'sex', 'race', 'ancestry', 'place of birth', 'citizenship',
        'education', 'employment status', 'marital status', 'religion',
        'disability', 'health insurance',
        'household language', 'household type', 'family presence and age'
    }

    features = []
    for key_value in persona.strip().split('\n'):
        if 'big five scores: ' in key_value:
            text_features = key_value[len('big five scores: '):]

            if text_features == 'Not Applicable':
                features.extend(['Not Applicable'] * 5)
            else:
                features.extend([
                    key_value.split(': ')[1]
                    for key_value in text_features.split(', ')
                ])
        else:
            key, value = key_value.split(': ')

            if key in feature_set:
                value = int(value) if key == 'age' else value
                features.append(value)

    return features
